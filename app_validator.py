"""Streamlit interface for validating extracted keywords."""

from __future__ import annotations

import ast
import hmac
import json
import os
import re
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd
from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, SKOS
import streamlit as st
from streamlit_scroll_to_top import scroll_to_here


APP_DIR = Path(__file__).parent
SOURCE_FILE = APP_DIR / "data"/ "keyword_extraction_review.xlsx"
LOCAL_SOURCE_FILE = APP_DIR / "data" / "keyword_extraction_review.csv"
VALIDATIONS_FILE = APP_DIR / "data" / "keyword_validations.json"
SOILVOC_FILE = APP_DIR / "data" / "SoilVoc.ttl"
SOILVOC_KEYWORDS_FILE = APP_DIR / "data" / "soilvoc_keywords.csv"
SOILVOC_URL = "https://raw.githubusercontent.com/soilwise-he/soil-vocabs/refs/heads/main/SoilVoc.ttl"
EUSOILVOC_PREFIX = "https://w3id.org/eusoilvoc#"
SOURCE_SHEET_PREFIX = "keyllm"
RESULTS_SOURCE_SHEET = "result"
FIXED_COLUMNS = ("record_identifier", "title", "abstract")
KEYWORD_COLUMNS = ("subject_label", "keyword_matcher_output", "concept_label")
KEYWORD_SOURCE_COLUMN = "keywords_source"
KEYWORD_SOURCES_FIELD = "keyword_sources"
KEYWORD_MATCHER_SOURCE = "keyword_matcher"
DELETED_KEYWORDS_FIELD = "deleted_keywords"
UNKNOWN_KEYWORD_SOURCE = "unknown_source"
VALIDATION_COLUMNS = KEYWORD_COLUMNS + ("new_keywords", "new_keyword_uris")
REMARKS_COLUMN = "general_remarks"
LEGACY_REMARKS_COLUMN = "General_remarks"
SCROLL_TO_TOP_FLAG = "scroll_to_top_on_render"
SCROLL_TO_TOP_TOKEN = "scroll_to_top_token"
SESSION_VALIDATIONS_KEY = "session_validations"
NON_HUMAN_USERNAMES = {
    "appuser",
    "unknown_user",
    "root",
    "admin",
    "ubuntu",
}
KEYWORD_LABELS = {
    "subject_label": "Source",
    "keyword_matcher_output": "Keyword matcher",
    "concept_label": "Keyword extraction",
    "new_keywords": "New keywords from SoilVoc",
}
DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
EXAMPLE_RECORD_IDENTIFIER = "__guided_keyword_review_example__"
EXAMPLE_RECORD = {
    "record_identifier": EXAMPLE_RECORD_IDENTIFIER,
    "title": "Welcome — guided keyword-review example",
    "abstract": (
        "Start here before reviewing the experiment records. Review each keyword list, "
        "remove suggestions that do not describe the record, and keep the relevant ones. "
        "In New keywords from SoilVoc, add 'erosion ratio'. "
        "Select Save this review to continue to the first source record."
    ),
    "subject_label": ["relevant source term", "example term to remove"],
    "keyword_matcher_output": ["relevant matcher result", "example mismatch"],
    "concept_label": ["example concept to remove", "relevant extracted concept"],
}
EXAMPLE_REQUIRED_EXISTING_KEYWORDS = {
    "subject_label": ["relevant source term"],
    "keyword_matcher_output": ["relevant matcher result"],
    "concept_label": ["relevant extracted concept"],
}
EXAMPLE_REQUIRED_NEW_KEYWORD = "erosion ratio"
GLOSIS_PROCEDURE_NAMESPACE = Namespace("http://w3id.org/glosis/model/procedure/")


def split_keywords(value: Any) -> list[str]:
    """Turn a cell value into clean keyword labels while retaining their order."""
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        if pd.isna(value):
            return []
        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = text.replace("\n", ";").split(";")
        values = parsed if isinstance(parsed, (list, tuple, set)) else [parsed]

    return [str(item).strip() for item in values if str(item).strip()]


def split_free_uris(value: str) -> list[str]:
    """Collect one or more user-entered URIs, one URI per line."""
    return list(dict.fromkeys(uri.strip() for uri in value.splitlines() if uri.strip()))


def normalize_keyword_for_column(column: str, keyword: str) -> str:
    """Normalize specific source keywords for UI readability and consistency."""
    if column == "keyword_matcher_output" and keyword.startswith(EUSOILVOC_PREFIX):
        return keyword[len(EUSOILVOC_PREFIX):].strip()
    return keyword


def normalize_keyword_list_for_column(column: str, keywords: list[str]) -> list[str]:
    """Normalize and de-duplicate a keyword list while preserving order."""
    normalized = [normalize_keyword_for_column(column, keyword) for keyword in keywords]
    return list(dict.fromkeys(keyword for keyword in normalized if keyword))


def doi_resolver_url(identifier: str) -> str | None:
    """Return a www.doi.org link when an identifier contains a DOI."""
    doi = re.sub(
        r"^(?:https?://)?(?:(?:dx|www)\.)?doi\.org/|^doi:\s*",
        "",
        identifier.strip(),
        flags=re.IGNORECASE,
    )
    if not DOI_PATTERN.fullmatch(doi):
        return None
    return f"https://www.doi.org/{quote(doi, safe='/')}"


def example_review_is_complete(kept_keywords: dict[str, list[str]]) -> bool:
    """Require the guided example's mismatches to be removed before continuing."""
    correct_existing_keywords = all(
        set(kept_keywords.get(column, [])) == set(required_keywords)
        for column, required_keywords in EXAMPLE_REQUIRED_EXISTING_KEYWORDS.items()
    )
    return (
        correct_existing_keywords
        and EXAMPLE_REQUIRED_NEW_KEYWORD in kept_keywords.get("new_keywords", [])
    )


def validate_source(
    data: pd.DataFrame,
    *,
    require_all_keyword_columns: bool = True,
    require_keyword_source_column: bool = False,
) -> pd.DataFrame:
    """Confirm that source data has the columns required by the review UI."""
    required_columns = set(FIXED_COLUMNS)
    if require_all_keyword_columns:
        required_columns.update(KEYWORD_COLUMNS)
    if require_keyword_source_column:
        required_columns.add(KEYWORD_SOURCE_COLUMN)
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"The source data is missing required columns: {missing}")

    if not require_all_keyword_columns:
        for column in KEYWORD_COLUMNS:
            if column not in data.columns:
                data[column] = ""

    if KEYWORD_SOURCE_COLUMN not in data.columns:
        data[KEYWORD_SOURCE_COLUMN] = ""

    return data


def first_present_value(values: pd.Series) -> Any:
    """Return the first non-empty source value, preserving the original value."""
    for value in values:
        if pd.notna(value) and str(value).strip():
            return value
    return ""


def aggregate_records(data: pd.DataFrame) -> pd.DataFrame:
    """Create one review record per identifier and collect all its keyword options."""
    aggregated_rows: list[dict[str, Any]] = []
    for record_identifier, record_rows in data.groupby("record_identifier", sort=False, dropna=False):
        aggregated_record: dict[str, Any] = {
            "record_identifier": record_identifier,
            "title": first_present_value(record_rows["title"]),
            "abstract": first_present_value(record_rows["abstract"]),
            KEYWORD_SOURCES_FIELD: {column: {} for column in KEYWORD_COLUMNS},
        }

        keywords_by_column: dict[str, list[str]] = {column: [] for column in KEYWORD_COLUMNS}
        keyword_sources_by_column: dict[str, dict[str, list[str]]] = {
            column: {} for column in KEYWORD_COLUMNS
        }

        for _, row in record_rows.iterrows():
            row_keyword_sources = split_keywords(row.get(KEYWORD_SOURCE_COLUMN, ""))

            for column in KEYWORD_COLUMNS:
                row_keywords = normalize_keyword_list_for_column(column, split_keywords(row[column]))
                keywords_by_column[column].extend(row_keywords)

                if column == "keyword_matcher_output":
                    for keyword in row_keywords:
                        source_list = keyword_sources_by_column[column].setdefault(keyword, [])
                        if KEYWORD_MATCHER_SOURCE not in source_list:
                            source_list.append(KEYWORD_MATCHER_SOURCE)
                    continue

                if not row_keywords:
                    continue

                if len(row_keyword_sources) == len(row_keywords):
                    keyword_source_pairs = zip(row_keywords, row_keyword_sources)
                elif len(row_keyword_sources) == 1:
                    keyword_source_pairs = (
                        (keyword, row_keyword_sources[0]) for keyword in row_keywords
                    )
                else:
                    keyword_source_pairs = ((keyword, "") for keyword in row_keywords)

                for keyword, source in keyword_source_pairs:
                    cleaned_source = str(source).strip()
                    if not cleaned_source:
                        continue
                    source_list = keyword_sources_by_column[column].setdefault(keyword, [])
                    if cleaned_source not in source_list:
                        source_list.append(cleaned_source)

        for column in KEYWORD_COLUMNS:
            unique_keywords = normalize_keyword_list_for_column(column, keywords_by_column[column])
            aggregated_record[column] = unique_keywords
            source_to_keywords: dict[str, list[str]] = {}
            for keyword in unique_keywords:
                for source in keyword_sources_by_column[column].get(keyword, []):
                    source_to_keywords.setdefault(source, []).append(keyword)
            aggregated_record[KEYWORD_SOURCES_FIELD][column] = {
                source: list(dict.fromkeys(keywords))
                for source, keywords in source_to_keywords.items()
            }

        aggregated_rows.append(aggregated_record)

    return pd.DataFrame(
        aggregated_rows,
        columns=FIXED_COLUMNS + KEYWORD_COLUMNS + (KEYWORD_SOURCES_FIELD,),
    )


def parse_keyword_sources(value: Any) -> dict[str, dict[str, list[str]]]:
    """Parse stored keyword source metadata as source-grouped keywords."""
    if isinstance(value, dict):
        raw_value = value
    elif pd.isna(value):
        return {column: {} for column in KEYWORD_COLUMNS}
    else:
        text = str(value).strip()
        if not text:
            return {column: {} for column in KEYWORD_COLUMNS}
        try:
            raw_value = json.loads(text)
        except json.JSONDecodeError:
            try:
                raw_value = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return {column: {} for column in KEYWORD_COLUMNS}

    if not isinstance(raw_value, dict):
        return {column: {} for column in KEYWORD_COLUMNS}

    parsed: dict[str, dict[str, list[str]]] = {column: {} for column in KEYWORD_COLUMNS}
    for column in KEYWORD_COLUMNS:
        source_map = raw_value.get(column, {})
        if not isinstance(source_map, dict):
            continue
        for source, keywords in source_map.items():
            if not str(source).strip():
                continue
            normalized_keywords = normalize_keyword_list_for_column(
                column,
                keywords if isinstance(keywords, list) else [keywords],
            )
            if normalized_keywords:
                parsed[column][str(source).strip()] = normalized_keywords
    return parsed


def selected_keyword_sources_for_record(
    record: pd.Series,
    kept_keywords: dict[str, list[str]],
) -> dict[str, dict[str, list[str]]]:
    """Return keyword-source metadata for all suggested keywords in a record."""
    del kept_keywords
    record_keyword_sources = parse_keyword_sources(record.get(KEYWORD_SOURCES_FIELD, {}))
    selected_sources: dict[str, dict[str, list[str]]] = {}

    for column in KEYWORD_COLUMNS:
        selected_sources[column] = {}
        available_keywords = normalize_keyword_list_for_column(
            column,
            split_keywords(record[column]),
        )
        available_set = set(available_keywords)

        if column == "keyword_matcher_output":
            if available_keywords:
                selected_sources[column][KEYWORD_MATCHER_SOURCE] = available_keywords
            continue

        for source, source_keywords in record_keyword_sources.get(column, {}).items():
            matched_keywords = [
                keyword for keyword in source_keywords if keyword in available_set
            ]
            if matched_keywords:
                selected_sources[column][source] = list(dict.fromkeys(matched_keywords))

    return selected_sources


def deleted_keywords_for_record(
    record: pd.Series,
    kept_keywords: dict[str, list[str]],
    keyword_sources: dict[str, dict[str, list[str]]],
) -> dict[str, dict[str, list[str]]]:
    """Capture removed keywords grouped by their source."""
    deleted: dict[str, dict[str, list[str]]] = {}
    for column in KEYWORD_COLUMNS:
        available_keywords = normalize_keyword_list_for_column(
            column,
            split_keywords(record[column]),
        )
        selected_keywords = set(
            normalize_keyword_list_for_column(column, kept_keywords.get(column, []))
        )
        deleted_keywords = [
            keyword for keyword in available_keywords if keyword not in selected_keywords
        ]
        deleted[column] = {}
        deleted_set = set(deleted_keywords)
        sourced_deleted: set[str] = set()

        for source, source_keywords in keyword_sources.get(column, {}).items():
            matches = [keyword for keyword in source_keywords if keyword in deleted_set]
            if matches:
                deleted[column][source] = list(dict.fromkeys(matches))
                sourced_deleted.update(matches)

        unsourced_deleted = [
            keyword for keyword in deleted_keywords if keyword not in sourced_deleted
        ]
        if unsourced_deleted:
            deleted[column][UNKNOWN_KEYWORD_SOURCE] = unsourced_deleted

        for source in list(deleted[column].keys()):
            deleted[column][source] = list(dict.fromkeys(deleted[column][source]))

    return deleted


def parse_deleted_keywords(value: Any) -> dict[str, dict[str, list[str]]]:
    """Parse deleted_keywords as source-grouped keywords."""
    if isinstance(value, dict):
        raw_value = value
    elif pd.isna(value):
        return {column: {} for column in KEYWORD_COLUMNS}
    else:
        text = str(value).strip()
        if not text:
            return {column: {} for column in KEYWORD_COLUMNS}
        try:
            raw_value = json.loads(text)
        except json.JSONDecodeError:
            try:
                raw_value = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return {column: {} for column in KEYWORD_COLUMNS}

    if not isinstance(raw_value, dict):
        return {column: {} for column in KEYWORD_COLUMNS}

    parsed: dict[str, dict[str, list[str]]] = {column: {} for column in KEYWORD_COLUMNS}
    for column in KEYWORD_COLUMNS:
        column_value = raw_value.get(column, {})
        if isinstance(column_value, dict):
            for source, keywords in column_value.items():
                parsed[column][str(source)] = normalize_keyword_list_for_column(
                    column,
                    keywords if isinstance(keywords, list) else [keywords],
                )

    return parsed


def load_workbook_records(workbook: Path | BytesIO) -> pd.DataFrame:
    """Load and combine all keyllm* sheets from an Excel workbook."""
    excel_file = pd.ExcelFile(workbook)
    source_sheets = [
        sheet_name
        for sheet_name in excel_file.sheet_names
        if sheet_name.casefold().startswith(SOURCE_SHEET_PREFIX)
    ]

    if RESULTS_SOURCE_SHEET in excel_file.sheet_names:
        source_sheets.append(RESULTS_SOURCE_SHEET)

    source_sheets = list(dict.fromkeys(source_sheets))

    if not source_sheets:
        raise ValueError(
            "The source workbook must contain at least one sheet whose name starts "
            f"with '{SOURCE_SHEET_PREFIX}' (or the general '{RESULTS_SOURCE_SHEET}' sheet)."
        )

    sheet_data = pd.read_excel(excel_file, sheet_name=source_sheets)
    frames: list[pd.DataFrame] = []
    for sheet_name in source_sheets:
        require_all_keyword_columns = sheet_name == RESULTS_SOURCE_SHEET
        try:
            frames.append(
                validate_source(
                    sheet_data[sheet_name],
                    require_all_keyword_columns=require_all_keyword_columns,
                    require_keyword_source_column=True,
                )
            )
        except ValueError as error:
            raise ValueError(f"Sheet '{sheet_name}' is invalid: {error}") from error


    return aggregate_records(pd.concat(frames, ignore_index=True))


@st.cache_data(show_spinner=False)
def load_workbook_records_cached(source_file_mtime_ns: int) -> pd.DataFrame:
    """Cache workbook parsing across reruns until the source file changes."""
    del source_file_mtime_ns
    return load_workbook_records(SOURCE_FILE)


def load_source() -> pd.DataFrame:
    """Prefer the Excel source and refresh the local CSV copy only when needed."""
    if SOURCE_FILE.exists():
        source_mtime_ns = SOURCE_FILE.stat().st_mtime_ns
        records = load_workbook_records_cached(source_mtime_ns)

        should_refresh_csv = True
        if LOCAL_SOURCE_FILE.exists():
            csv_mtime_ns = LOCAL_SOURCE_FILE.stat().st_mtime_ns
            should_refresh_csv = csv_mtime_ns < source_mtime_ns

        if should_refresh_csv:
            LOCAL_SOURCE_FILE.parent.mkdir(parents=True, exist_ok=True)
            records.to_csv(LOCAL_SOURCE_FILE, index=False)
    elif LOCAL_SOURCE_FILE.exists():
        records = validate_source(pd.read_csv(LOCAL_SOURCE_FILE))
    else:
        raise FileNotFoundError(
            "Add keyword_extraction_review.xlsx beside this app, or add "
            "data/keyword_extraction_review.csv."
        )

    return records


def import_source(uploaded_file: Any) -> int:
    """Validate and save an administrator-uploaded workbook and its CSV copy."""
    workbook = uploaded_file.getvalue()
    records = load_workbook_records(BytesIO(workbook))
    SOURCE_FILE.write_bytes(workbook)
    LOCAL_SOURCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(LOCAL_SOURCE_FILE, index=False)
    return len(records)


def soilvoc_labels(turtle_data: bytes) -> list[str]:
    """Extract non-procedure concept labels from a SoilVoc Turtle file."""
    graph = Graph()
    graph.parse(data=turtle_data, format="turtle")

    procedure_namespace = str(GLOSIS_PROCEDURE_NAMESPACE)

    def is_procedure_link(node: Any) -> bool:
        return isinstance(node, URIRef) and str(node).startswith(procedure_namespace)

    def concept_is_procedure_related(concept: Any) -> bool:
        for _, obj in graph.predicate_objects(concept):
            if is_procedure_link(obj):
                return True
            if isinstance(obj, BNode):
                for _, nested_obj in graph.predicate_objects(obj):
                    if is_procedure_link(nested_obj):
                        return True
        return False

    labels = set()
    for concept in graph.subjects(predicate=RDF.type, object=SKOS.Concept):
        if concept_is_procedure_related(concept):
            continue
        for predicate in (SKOS.prefLabel, RDFS.label):
            for label in graph.objects(subject=concept, predicate=predicate):
                if isinstance(label, Literal):
                    text = str(label).strip()
                    if text:
                        labels.add(text)

    if not labels:
        raise ValueError("The downloaded SoilVoc file contains no concept labels.")
    return sorted(labels, key=str.casefold)


def ingest_soilvoc() -> int:
    """Download, validate, and store the latest SoilVoc labels locally."""
    try:
        with urlopen(SOILVOC_URL, timeout=30) as response:
            turtle_data = response.read()
    except (OSError, URLError) as error:
        raise ValueError(f"SoilVoc could not be downloaded: {error}") from error

    try:
        labels = soilvoc_labels(turtle_data)
    except Exception as error:
        raise ValueError(f"The downloaded SoilVoc file is invalid: {error}") from error

    SOILVOC_FILE.parent.mkdir(parents=True, exist_ok=True)
    SOILVOC_FILE.write_bytes(turtle_data)
    pd.DataFrame({"keyword": labels}).to_csv(SOILVOC_KEYWORDS_FILE, index=False)
    return len(labels)


def load_soilvoc_keywords() -> list[str]:
    """Load locally indexed SoilVoc labels for adding new keywords."""
    # Prefer deriving from the local Turtle vocabulary so filtering stays semantic.
    if SOILVOC_FILE.exists():
        try:
            labels = soilvoc_labels(SOILVOC_FILE.read_bytes())
        except (OSError, ValueError):
            labels = []
        else:
            SOILVOC_KEYWORDS_FILE.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"keyword": labels}).to_csv(SOILVOC_KEYWORDS_FILE, index=False)
            return labels

    if SOILVOC_KEYWORDS_FILE.exists():
        try:
            csv_data = pd.read_csv(SOILVOC_KEYWORDS_FILE)
            if "keyword" in csv_data.columns:
                keywords = csv_data["keyword"].dropna().astype(str).str.strip()
                keywords = keywords[keywords != ""]
                if not keywords.empty:
                    return keywords.tolist()
        except (OSError, ValueError, pd.errors.EmptyDataError):
            pass

    return []


def get_admin_password() -> str | None:
    """Load the password from an environment variable or Streamlit secrets."""
    if password := os.getenv("KEYWORD_VALIDATOR_ADMIN_PASSWORD"):
        return password
    try:
        password = st.secrets.get("admin_password", "")
    except FileNotFoundError:
        return None
    return str(password) or None


def get_default_username() -> str:
    """Return a sensible default username for exported review metadata."""
    # 1) Prefer explicit app-level overrides.
    try:
        configured = str(st.secrets.get("default_username", "")).strip()
    except FileNotFoundError:
        configured = ""
    if configured:
        return configured

    # 2) Allow pre-filling through URL query params, e.g. ?username=alice.
    try:
        query_username = str(st.query_params.get("username", "")).strip()
    except Exception:
        query_username = ""
    if query_username:
        return query_username

    # 3) Use logged-in Streamlit user metadata when available.
    streamlit_user = getattr(st, "user", None)
    if streamlit_user:
        for field_name in ("name", "email"):
            try:
                value = str(getattr(streamlit_user, field_name, "")).strip()
            except Exception:
                value = ""
            if value:
                return value

    # 4) Fall back to environment and OS login, but skip service-account values.
    for env_name in ("USERNAME", "USER", "LOGNAME"):
        value = os.getenv(env_name)
        if value and value.strip() and value.strip().casefold() not in NON_HUMAN_USERNAMES:
            return value.strip()
    try:
        login_name = os.getlogin().strip()
        if login_name.casefold() not in NON_HUMAN_USERNAMES:
            return login_name
    except OSError:
        pass

    # Last resort: keep empty so the reviewer can provide a real identifier.
    return ""


def render_sidebar_download_button(
    validations: dict[str, dict[str, Any]],
    has_unsaved_changes: bool = False,
) -> None:
    """Provide a red download button in the sidebar with review metadata."""
    if "session_started_at" not in st.session_state:
        st.session_state.session_started_at = datetime.now().isoformat()

    with st.sidebar:
        st.divider()
        st.subheader("Export")
        st.caption(
            "The keyword matching team (Jiarong, Beichen, Paul, and Max) thanks you for your effort 🙏"
        )
        username_input = st.text_input(
            "Username",
            value=st.session_state.get("reviewer_username", get_default_username()),
            key="reviewer_username",
        )
        username = username_input.strip()
        metadata = {
            "username": username,
            "start_session": st.session_state.session_started_at,
            "download_time": datetime.now().isoformat(),
        }
        payload = {
            "metadata": metadata,
            "validations": validations,
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"keyword_validations_{timestamp}_{username}.json"
        username_missing = not username
        st.markdown(
            """
            <style>
            div[data-testid="stSidebar"] div.stDownloadButton > button {
                background-color: #d32f2f;
                border-color: #d32f2f;
                color: white;
            }
            div[data-testid="stSidebar"] div.stDownloadButton > button:hover {
                background-color: #b71c1c;
                border-color: #b71c1c;
                color: white;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        if username_missing:
            st.caption("Enter a username to enable download.")
        elif has_unsaved_changes:
            st.caption("Save this record first to include your latest changes in the export.")

        button_disabled = username_missing or has_unsaved_changes
        if username_missing:
            button_help = "Add a username above to enable this download."
        elif has_unsaved_changes:
            button_help = "Save this record first to include all current changes."
        else:
            button_help = "Download the current session validations as JSON."

        st.download_button(
            "📁 Download validations",
            data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=filename,
            mime="application/json",
            key="download_validations_button",
            type="primary",
            disabled=button_disabled,
            help=button_help,
        )


def render_admin_import() -> None:
    """Show the replacement-source uploader only after administrator sign-in."""
    configured_password = get_admin_password()
    with st.sidebar:
        st.divider()
        st.subheader("Administrator")
        if not configured_password:
            st.info("Workbook import is disabled until an administrator password is configured.")
            return

        if not st.session_state.get("is_admin", False):
            with st.form("admin_sign_in"):
                entered_password = st.text_input("Administrator password", type="password")
                sign_in = st.form_submit_button("Sign in")
            if sign_in:
                if hmac.compare_digest(entered_password, configured_password):
                    st.session_state.is_admin = True
                    st.rerun()
                st.error("Incorrect administrator password.")
            return

        if st.button("Sign out"):
            st.session_state.is_admin = False
            st.rerun()

        if st.button("Refresh SoilVoc keyword source"):
            try:
                keyword_count = ingest_soilvoc()
            except ValueError as error:
                st.error(str(error))
            else:
                st.success(f"Stored {keyword_count} SoilVoc labels locally.")

        uploaded_file = st.file_uploader(
            "Replace source workbook",
            type=["xlsx"],
            help=(
                "The workbook must contain at least one sheet starting with 'keyllm' "
                "(or the legacy 'result' sheet) and the required columns."
            ),
        )
        if uploaded_file and st.button("Import workbook", type="primary"):
            try:
                record_count = import_source(uploaded_file)
            except (OSError, ValueError) as error:
                st.error(f"The workbook was not imported: {error}")
            else:
                st.session_state.record_position = 0
                st.success(f"Imported {record_count} records. Existing reviews were retained.")
                st.rerun()

def load_validations() -> dict[str, dict[str, Any]]:
    """Load existing review decisions keyed by record identifier."""
    if VALIDATIONS_FILE.exists():
        with VALIDATIONS_FILE.open("r", encoding="utf-8") as validation_file:
            raw_data = json.load(validation_file)

        if isinstance(raw_data, dict) and "validations" in raw_data:
            raw_data = raw_data["validations"]

        validations: dict[str, dict[str, Any]] = {}
        for record_id in raw_data:
            record_data = raw_data.get(record_id)
            if not isinstance(record_data, dict):
                continue

            loaded_record = {
                column: normalize_keyword_list_for_column(
                    column,
                    record_data.get(column, []) if isinstance(record_data.get(column, []), list) else [],
                )
                for column in VALIDATION_COLUMNS
            }
            loaded_record[REMARKS_COLUMN] = str(
                record_data.get(REMARKS_COLUMN, record_data.get(LEGACY_REMARKS_COLUMN, ""))
            )
            loaded_record[KEYWORD_SOURCES_FIELD] = parse_keyword_sources(
                record_data.get(KEYWORD_SOURCES_FIELD, {})
            )
            loaded_record[DELETED_KEYWORDS_FIELD] = parse_deleted_keywords(
                record_data.get(DELETED_KEYWORDS_FIELD, {})
            )
            validations[record_id] = loaded_record
        return validations

    legacy_validation_file = APP_DIR / "data" / "keyword_validations.csv"
    if not legacy_validation_file.exists():
        return {}

    saved = pd.read_csv(legacy_validation_file, dtype={"record_identifier": str})
    validations: dict[str, dict[str, Any]] = {}
    for row in saved.itertuples(index=False):
        kept_keywords = json.loads(row.kept_keywords)
        if isinstance(kept_keywords, list):
            validations[row.record_identifier] = {
                column: normalize_keyword_list_for_column(column, kept_keywords)
                for column in KEYWORD_COLUMNS
            }
            validations[row.record_identifier][KEYWORD_SOURCES_FIELD] = {}
            validations[row.record_identifier][DELETED_KEYWORDS_FIELD] = {
                column: {} for column in KEYWORD_COLUMNS
            }
            validations[row.record_identifier]["new_keywords"] = []
            validations[row.record_identifier]["new_keyword_uris"] = []
            validations[row.record_identifier][REMARKS_COLUMN] = ""
        else:
            validations[row.record_identifier] = {
                column: normalize_keyword_list_for_column(
                    column,
                    kept_keywords.get(column, []) if isinstance(kept_keywords.get(column, []), list) else [],
                )
                for column in VALIDATION_COLUMNS
            }
            validations[row.record_identifier][KEYWORD_SOURCES_FIELD] = parse_keyword_sources(
                kept_keywords.get(KEYWORD_SOURCES_FIELD, {})
            )
            validations[row.record_identifier][DELETED_KEYWORDS_FIELD] = parse_deleted_keywords(
                kept_keywords.get(DELETED_KEYWORDS_FIELD, {})
            )
            validations[row.record_identifier][REMARKS_COLUMN] = str(
                kept_keywords.get(REMARKS_COLUMN, kept_keywords.get(LEGACY_REMARKS_COLUMN, ""))
            )
    return validations


def save_validations(validations: dict[str, dict[str, Any]]) -> None:
    """Persist the keywords that remain after each review decision as JSON."""
    VALIDATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with VALIDATIONS_FILE.open("w", encoding="utf-8") as validation_file:
        json.dump(validations, validation_file, ensure_ascii=False, indent=2)


def expected_keywords_for_record(
    record: pd.Series,
    record_id: str,
    saved_keywords: dict[str, Any],
    soilvoc_keywords: list[str],
) -> dict[str, Any]:
    """Compute the persisted/default selections for a record."""
    expected: dict[str, Any] = {}
    for source_column in KEYWORD_COLUMNS:
        available_keywords = normalize_keyword_list_for_column(
            source_column,
            split_keywords(record[source_column]),
        )
        default_keywords = saved_keywords.get(source_column, available_keywords)
        default_keywords = normalize_keyword_list_for_column(source_column, default_keywords)
        expected[source_column] = [
            keyword for keyword in default_keywords if keyword in available_keywords
        ]

    if soilvoc_keywords:
        expected["new_keywords"] = [
            keyword
            for keyword in saved_keywords.get("new_keywords", [])
            if keyword in soilvoc_keywords
        ]
    else:
        expected["new_keywords"] = saved_keywords.get("new_keywords", [])

    expected["new_keyword_uris"] = split_free_uris(
        "\n".join(saved_keywords.get("new_keyword_uris", []))
    )
    expected[REMARKS_COLUMN] = str(saved_keywords.get(REMARKS_COLUMN, ""))
    return expected


def current_keywords_from_session(
    record: pd.Series,
    record_id: str,
    saved_keywords: dict[str, Any],
    soilvoc_keywords: list[str],
) -> dict[str, Any]:
    """Read current widget selections, falling back to current defaults when untouched."""
    expected = expected_keywords_for_record(record, record_id, saved_keywords, soilvoc_keywords)
    current: dict[str, Any] = {}

    for source_column in KEYWORD_COLUMNS:
        widget_key = f"keywords_{source_column}_{record_id}"
        current[source_column] = st.session_state.get(widget_key, expected[source_column])

    new_keywords_key = f"keywords_new_keywords_{record_id}"
    current["new_keywords"] = st.session_state.get(new_keywords_key, expected["new_keywords"])

    uri_widget_key = f"keywords_new_keyword_uris_{record_id}"
    uri_widget_value = st.session_state.get(
        uri_widget_key,
        "\n".join(expected["new_keyword_uris"]),
    )
    current["new_keyword_uris"] = split_free_uris(uri_widget_value)

    remarks_widget_key = f"keywords_{REMARKS_COLUMN}_{record_id}"
    current[REMARKS_COLUMN] = str(
        st.session_state.get(remarks_widget_key, expected[REMARKS_COLUMN])
    )
    return current


def has_unsaved_record_changes(
    record: pd.Series,
    record_id: str,
    saved_keywords: dict[str, Any],
    soilvoc_keywords: list[str],
) -> bool:
    """Check whether in-memory widget selections differ from persisted/default values."""
    expected = expected_keywords_for_record(record, record_id, saved_keywords, soilvoc_keywords)
    current = current_keywords_from_session(record, record_id, saved_keywords, soilvoc_keywords)

    for column in VALIDATION_COLUMNS:
        if set(current.get(column, [])) != set(expected.get(column, [])):
            return True
    if current.get(REMARKS_COLUMN, "") != expected.get(REMARKS_COLUMN, ""):
        return True
    return False


def request_scroll_to_top() -> None:
    """Request a scroll-to-top action for the current or next render."""
    st.session_state[SCROLL_TO_TOP_FLAG] = True
    st.session_state[SCROLL_TO_TOP_TOKEN] = st.session_state.get(SCROLL_TO_TOP_TOKEN, 0) + 1


def main() -> None:
    st.set_page_config(page_title="Keyword experiment validator", layout="wide")

    if st.session_state.get(SCROLL_TO_TOP_FLAG):
        # Use the dedicated component's iframe focus workaround, as recommended in
        # the Streamlit discussion thread.
        scroll_to_here(0, key=f"scroll_to_top_{st.session_state.get(SCROLL_TO_TOP_TOKEN, 0)}")
        st.session_state[SCROLL_TO_TOP_FLAG] = False

    st.title("Keyword experiment validator")
    st.caption("Review one record at a time. Remove any keyword that should not be retained.")
    render_admin_import()

    try:
        records = load_source()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.info(
            "The Excel source is expected at keyword_extraction_review.xlsx, with at "
            "least one sheet starting with 'keyllm' (or the legacy 'result' sheet). "
            "When present, the app creates a reusable local CSV copy."
        )
        return
    
    
    if records.empty:
        st.warning("The source contains no records to review.")
        return

    records = records.copy()
    records["record_identifier"] = records["record_identifier"].astype(str)
    validations = load_validations()
    if SESSION_VALIDATIONS_KEY not in st.session_state:
        st.session_state[SESSION_VALIDATIONS_KEY] = {}

    record_identifiers = records["record_identifier"].tolist()
    if set(st.session_state.get("record_order", [])) != set(record_identifiers):
        st.session_state.record_order = records.sample(frac=1)["record_identifier"].tolist()
        st.session_state.record_position = 0
    records = (
        records.set_index("record_identifier")
        .loc[st.session_state.record_order]
        .reset_index()
    )
    records = pd.concat([pd.DataFrame([EXAMPLE_RECORD]), records], ignore_index=True)

    if "record_position" not in st.session_state:
        st.session_state.record_position = 0
    st.session_state.record_position = min(st.session_state.record_position, len(records) - 1)

    position = st.session_state.record_position
    record = records.iloc[position]
    record_id = record["record_identifier"]
    saved_keywords = validations.get(record_id, {})
    soilvoc_keywords = load_soilvoc_keywords()
    has_unsaved_changes = has_unsaved_record_changes(
        record,
        record_id,
        saved_keywords,
        soilvoc_keywords,
    )
    render_sidebar_download_button(
        st.session_state[SESSION_VALIDATIONS_KEY],
        has_unsaved_changes=has_unsaved_changes,
    )

    if "pending_next_confirmation_record_id" not in st.session_state:
        st.session_state.pending_next_confirmation_record_id = None

    is_example_record = record_id == EXAMPLE_RECORD_IDENTIFIER
    next_disabled = position == len(records) - 1 or is_example_record
    next_help = None
    if is_example_record:
        next_help = "Complete and save the tutorial example first to unlock Next."
    elif position == len(records) - 1:
        next_help = "You are already at the last record."

    previous_column, progress_column, next_column = st.columns([1, 3, 1])
    with previous_column:
        if st.button("← Previous", disabled=position == 0):
            st.session_state.pending_next_confirmation_record_id = None
            request_scroll_to_top()
            st.session_state.record_position -= 1
            st.rerun()
    with progress_column:
        st.progress((position + 1) / len(records), text=f"Record {position + 1} of {len(records)}")
    with next_column:
        if st.button("Next →", disabled=next_disabled, help=next_help):
            if has_unsaved_changes:
                st.session_state.pending_next_confirmation_record_id = record_id
            else:
                st.session_state.pending_next_confirmation_record_id = None
                request_scroll_to_top()
                st.session_state.record_position += 1
                st.rerun()

    if st.session_state.pending_next_confirmation_record_id == record_id:
        if hasattr(st, "dialog"):
            @st.dialog("Unsaved changes")
            def confirm_next_navigation() -> None:
                st.write(
                    "You have unsaved changes for this record. Moving to the next record "
                    "will discard them. Are you sure you want to continue?"
                )
                st.caption("💾 Save button on the bottom of the page.")
                discard_column, stay_column = st.columns(2)
                with discard_column:
                    if st.button("Discard changes and continue"):
                        st.session_state.pending_next_confirmation_record_id = None
                        request_scroll_to_top()
                        st.session_state.record_position += 1
                        st.rerun()
                with stay_column:
                    if st.button("Stay on this record", type="primary"):
                        st.session_state.pending_next_confirmation_record_id = None
                        st.rerun()

            confirm_next_navigation()
        else:
            with st.container(border=True):
                st.warning(
                    "You have unsaved changes for this record. Moving to the next record will "
                    "discard them. Are you sure you want to continue?"
                )
                discard_column, stay_column = st.columns(2)
                with discard_column:
                    if st.button("Discard changes and continue"):
                        st.session_state.pending_next_confirmation_record_id = None
                        request_scroll_to_top()
                        st.session_state.record_position += 1
                        st.rerun()
                with stay_column:
                    if st.button("Stay on this record", type="primary"):
                        st.session_state.pending_next_confirmation_record_id = None
                        st.rerun()

    if is_example_record:
        st.info(
            "⚠️ Complete this tutorial to get started. ⚠️ "
            "\n\n"
            "**To-do:** remove every illustrative mismatch and add **erosion ratio** "
            "in missing keywords from SoilVoc, then select **Save this review**. This example "
            "is not written to the validation file."
        )

    st.subheader("Source record")
    with st.container(border=True):
        st.markdown("**Record identifier**")
        if doi_url := doi_resolver_url(record_id):
            # st.link_button(record_id, doi_url)
            st.write(doi_url)
        else:
            st.write(record_id)
        st.markdown("**Title**")
        st.write(str(record["title"]))
        st.markdown("**Abstract**")
        st.write(str(record["abstract"]))

    st.subheader("Automated results review")
    st.caption("Remove an item from the lists below if you consider it to be irrelevant or incorrect.")
    kept_keywords: dict[str, list[str]] = {}
    for source_column in KEYWORD_COLUMNS:
        available_keywords = normalize_keyword_list_for_column(
            source_column,
            split_keywords(record[source_column]),
        )
        default_keywords = saved_keywords.get(source_column, available_keywords)
        default_keywords = normalize_keyword_list_for_column(source_column, default_keywords)
        default_keywords = [
            keyword for keyword in default_keywords if keyword in available_keywords
        ]
        with st.container(border=True,):
            kept_keywords[source_column] = st.multiselect(
                KEYWORD_LABELS[source_column],
                label_visibility="collapsed",
                options=available_keywords,
                default=default_keywords,
                key=f"keywords_{source_column}_{record_id}",
            )

    st.subheader("Adding missing keywords")
    st.caption("Add keyword(s) from the drop-down list sourced from SoilVoc, or supplement the suggestions with URIs (new line separated) from other vocabularies")
    with st.container(border=True):
        if soilvoc_keywords:
            kept_keywords["new_keywords"] = st.multiselect(
                KEYWORD_LABELS["new_keywords"],
                options=soilvoc_keywords,
                default=saved_keywords.get("new_keywords", []),
                key=f"keywords_new_keywords_{record_id}",
                help="Select additional concepts from the locally imported SoilVoc vocabulary.",
            )
        else:
            st.info("An administrator can refresh the SoilVoc keyword source from the sidebar.")
            kept_keywords["new_keywords"] = saved_keywords.get("new_keywords", [])

        free_uri_text = st.text_area(
            "Free URIs",
            value="\n".join(saved_keywords.get("new_keyword_uris", [])),
            height=100,
            key=f"keywords_new_keyword_uris_{record_id}",
            help="Add custom concept URIs not available in SoilVoc, one URI per line.",
        )
        kept_keywords["new_keyword_uris"] = split_free_uris(free_uri_text)

    with st.expander("Add more remarks", expanded=False):

        st.caption("Optional: add any notes about this record or its keywords.")

        free_remarks = st.text_area(
                            "Remarks",
                            value=str(saved_keywords.get(REMARKS_COLUMN, "")),
                            height=100,
                            key=f"keywords_{REMARKS_COLUMN}_{record_id}",
                        )
        kept_keywords[REMARKS_COLUMN] = free_remarks

    if st.button("💾 Save this review and go to the next record ⏭️", type="primary",width='stretch'):
        example_complete = not is_example_record or example_review_is_complete(kept_keywords)
        if not example_complete:
            st.error(
                "To continue, remove all example mismatches and add 'erosion ratio' "
                "under Adding missing keywords from SoilVoc."
            )
        elif not is_example_record:
            kept_keywords[KEYWORD_SOURCES_FIELD] = selected_keyword_sources_for_record(
                record,
                kept_keywords,
            )
            kept_keywords[DELETED_KEYWORDS_FIELD] = deleted_keywords_for_record(
                record,
                kept_keywords,
                kept_keywords[KEYWORD_SOURCES_FIELD],
            )
            validations[record_id] = kept_keywords
            st.session_state[SESSION_VALIDATIONS_KEY][record_id] = kept_keywords
            save_validations(validations)
        if example_complete and position < len(records) - 1:
            request_scroll_to_top()
            st.session_state.record_position += 1
            st.rerun()
        if example_complete and is_example_record:
            request_scroll_to_top()
            st.success("Example complete.")
        elif example_complete:
            request_scroll_to_top()
            st.success("Review saved. This is the last record.")

if __name__ == "__main__":
    main()
