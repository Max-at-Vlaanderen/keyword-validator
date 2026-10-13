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
import streamlit.components.v1 as components


APP_DIR = Path(__file__).parent
SOURCE_FILE = APP_DIR / "data"/ "keyword_extraction_review.xlsx"
LOCAL_SOURCE_FILE = APP_DIR / "data" / "keyword_extraction_review.csv"
VALIDATIONS_FILE = APP_DIR / "data" / "keyword_validations.json"
SOILVOC_FILE = APP_DIR / "data" / "SoilVoc.ttl"
SOILVOC_KEYWORDS_FILE = APP_DIR / "data" / "soilvoc_keywords.csv"
SOILVOC_URL = "https://raw.githubusercontent.com/soilwise-he/soil-vocabs/refs/heads/main/SoilVoc.ttl"
EUSOILVOC_PREFIX = "https://w3id.org/eusoilvoc#"
SOURCE_SHEET = "result"
FIXED_COLUMNS = ("record_identifier", "title", "abstract")
KEYWORD_COLUMNS = ("subject_label", "keyword_matcher_output", "concept_label")
VALIDATION_COLUMNS = KEYWORD_COLUMNS + ("new_keywords", "new_keyword_uris")
REMARKS_COLUMN = "general_remarks"
LEGACY_REMARKS_COLUMN = "General_remarks"
SCROLL_TO_TOP_FLAG = "scroll_to_top_on_render"
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


def validate_source(data: pd.DataFrame) -> pd.DataFrame:
    """Confirm that the source data has the columns required by the review UI."""
    required_columns = set(FIXED_COLUMNS + KEYWORD_COLUMNS)
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"The source data is missing required columns: {missing}")
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
        }
        for column in KEYWORD_COLUMNS:
            keywords: list[str] = []
            for value in record_rows[column]:
                keywords.extend(split_keywords(value))
            aggregated_record[column] = normalize_keyword_list_for_column(column, keywords)
        aggregated_rows.append(aggregated_record)

    return pd.DataFrame(aggregated_rows, columns=FIXED_COLUMNS + KEYWORD_COLUMNS)


def load_source() -> pd.DataFrame:
    """Prefer the Excel source, then retain a local CSV copy for future sessions."""
    if SOURCE_FILE.exists():
        data = pd.read_excel(SOURCE_FILE, sheet_name=SOURCE_SHEET)
    elif LOCAL_SOURCE_FILE.exists():
        data = pd.read_csv(LOCAL_SOURCE_FILE)
    else:
        raise FileNotFoundError(
            "Add keyword_extraction_review.xlsx beside this app, or add "
            "data/keyword_extraction_review.csv."
        )

    records = aggregate_records(validate_source(data))
    LOCAL_SOURCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(LOCAL_SOURCE_FILE, index=False)
    return records


def import_source(uploaded_file: Any) -> int:
    """Validate and save an administrator-uploaded workbook and its CSV copy."""
    workbook = uploaded_file.getvalue()
    records = aggregate_records(
        validate_source(pd.read_excel(BytesIO(workbook), sheet_name=SOURCE_SHEET))
    )
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


def render_sidebar_download_button(validations: dict[str, dict[str, Any]]) -> None:
    """Provide a red download button in the sidebar with review metadata."""
    if "session_started_at" not in st.session_state:
        st.session_state.session_started_at = datetime.now().isoformat()

    with st.sidebar:
        st.divider()
        st.subheader("Export")
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
        st.download_button(
            "📁 Download validations",
            data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=filename,
            mime="application/json",
            key="download_validations_button",
            type="primary",
            disabled=username_missing,
            help="Add a username above to enable this download.",
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
            help="The workbook must contain the 'result' sheet and the required columns.",
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


def render_scroll_to_top_if_requested() -> None:
    """Run a small script to move the app viewport to the top."""
    if st.session_state.get(SCROLL_TO_TOP_FLAG):
        components.html(
            """
            <script>
            const target = window.parent || window;
            target.scrollTo({ top: 0, left: 0, behavior: 'auto' });
            </script>
            """,
            height=0,
        )
        st.session_state[SCROLL_TO_TOP_FLAG] = False


def main() -> None:
    st.set_page_config(page_title="Keyword experiment validator", layout="wide")
    st.title("Keyword experiment validator")
    st.caption("Review one record at a time. Remove any keyword that should not be retained.")
    render_scroll_to_top_if_requested()
    render_admin_import()

    try:
        records = load_source()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.info(
            "The Excel source is expected at keyword_extraction_review.xlsx, on the "
            "'result' sheet. When present, the app creates a reusable local CSV copy."
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
    render_sidebar_download_button(st.session_state[SESSION_VALIDATIONS_KEY])

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
            st.session_state.record_position -= 1
            st.rerun()
    with progress_column:
        st.progress((position + 1) / len(records), text=f"Record {position + 1} of {len(records)}")
    with next_column:
        if st.button("Next →", disabled=next_disabled, help=next_help):
            if has_unsaved_record_changes(record, record_id, saved_keywords, soilvoc_keywords):
                st.session_state.pending_next_confirmation_record_id = record_id
            else:
                st.session_state.pending_next_confirmation_record_id = None
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
                    if st.button("Discard changes and continue", type="primary"):
                        st.session_state.pending_next_confirmation_record_id = None
                        st.session_state.record_position += 1
                        st.rerun()
                with stay_column:
                    if st.button("Stay on this record"):
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
                    if st.button("Discard changes and continue", type="primary"):
                        st.session_state.pending_next_confirmation_record_id = None
                        st.session_state.record_position += 1
                        st.rerun()
                with stay_column:
                    if st.button("Stay on this record"):
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
        with st.container(border=True):
            kept_keywords[source_column] = st.multiselect(
                KEYWORD_LABELS[source_column],
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

    if st.button("💾 Save this review and go to the next record ⏭️", type="primary",use_container_width=True):
        example_complete = not is_example_record or example_review_is_complete(kept_keywords)
        if not example_complete:
            st.error(
                "To continue, remove all example mismatches and add 'erosion ratio' "
                "under Adding missing keywords from SoilVoc."
            )
        elif not is_example_record:
            validations[record_id] = kept_keywords
            st.session_state[SESSION_VALIDATIONS_KEY][record_id] = kept_keywords
            save_validations(validations)
        if example_complete and position < len(records) - 1:
            request_scroll_to_top()
            st.session_state.record_position += 1
            st.rerun()
        if example_complete and is_example_record:
            request_scroll_to_top()
            render_scroll_to_top_if_requested()
            st.success("Example complete.")
        elif example_complete:
            request_scroll_to_top()
            render_scroll_to_top_if_requested()
            st.success("Review saved. This is the last record.")



if __name__ == "__main__":
    main()
