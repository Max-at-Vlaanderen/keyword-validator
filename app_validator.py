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
from rdflib import Graph, Literal
from rdflib.namespace import RDFS, SKOS
import streamlit as st


APP_DIR = Path(__file__).parent
SOURCE_FILE = APP_DIR / "keyword_extraction_review.xlsx"
LOCAL_SOURCE_FILE = APP_DIR / "data" / "keyword_extraction_review.csv"
VALIDATIONS_FILE = APP_DIR / "data" / "keyword_validations.json"
SOILVOC_FILE = APP_DIR / "data" / "SoilVoc.ttl"
SOILVOC_KEYWORDS_FILE = APP_DIR / "data" / "soilvoc_keywords.csv"
SOILVOC_URL = "https://raw.githubusercontent.com/soilwise-he/soil-vocabs/refs/heads/main/SoilVoc.ttl"
SOURCE_SHEET = "result"
FIXED_COLUMNS = ("record_identifier", "title", "abstract")
KEYWORD_COLUMNS = ("subject_label", "keyword_matcher_output", "concept_label")
VALIDATION_COLUMNS = KEYWORD_COLUMNS + ("new_keywords", "new_keyword_uris")
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
    "concept_label": ["relevant extracted concept", "example concept to remove"],
}
EXAMPLE_REQUIRED_EXISTING_KEYWORDS = {
    "subject_label": ["relevant source term"],
    "keyword_matcher_output": ["relevant matcher result"],
    "concept_label": ["relevant extracted concept"],
}
EXAMPLE_REQUIRED_NEW_KEYWORD = "erosion ratio"


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
            aggregated_record[column] = list(dict.fromkeys(keywords))
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
    """Extract unique human-readable concept labels from a SoilVoc Turtle file."""
    graph = Graph()
    graph.parse(data=turtle_data, format="turtle")
    labels = {
        str(label).strip()
        for predicate in (SKOS.prefLabel, RDFS.label)
        for label in graph.objects(predicate=predicate)
        if isinstance(label, Literal) and str(label).strip()
    }
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
    if SOILVOC_KEYWORDS_FILE.exists():
        return pd.read_csv(SOILVOC_KEYWORDS_FILE)["keyword"].dropna().tolist()
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
    for env_name in ("USERNAME", "USER", "LOGNAME"):
        value = os.getenv(env_name)
        if value and value.strip():
            return value.strip()
    try:
        return os.getlogin().strip()
    except OSError:
        return "unknown_user"


def render_sidebar_download_button(validations: dict[str, dict[str, list[str]]]) -> None:
    """Provide a red download button in the sidebar with review metadata."""
    if "session_started_at" not in st.session_state:
        st.session_state.session_started_at = datetime.now(timezone.utc).isoformat()

    with st.sidebar:
        st.divider()
        st.subheader("Export")
        username = st.text_input(
            "Username",
            value=st.session_state.get("reviewer_username", get_default_username()),
            key="reviewer_username",
        )
        metadata = {
            "username": username,
            "start_session": st.session_state.session_started_at,
            "download_time": datetime.now(timezone.utc).isoformat(),
        }
        payload = {
            "metadata": metadata,
            "validations": validations,
        }
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"keyword_validations_{timestamp}_{username or 'user'}.json"
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
        st.download_button(
            "Download validations",
            data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=filename,
            mime="application/json",
            key="download_validations_button",
            type="primary",
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

def load_validations() -> dict[str, dict[str, list[str]]]:
    """Load existing review decisions keyed by record identifier."""
    if VALIDATIONS_FILE.exists():
        with VALIDATIONS_FILE.open("r", encoding="utf-8") as validation_file:
            raw_data = json.load(validation_file)

        if isinstance(raw_data, dict) and "validations" in raw_data:
            raw_data = raw_data["validations"]

        return {
            record_id: {
                column: value if isinstance(value, list) else []
                for column in VALIDATION_COLUMNS
                for value in [raw_data.get(record_id, {}).get(column, [])]
            }
            for record_id in raw_data
            if isinstance(raw_data.get(record_id), dict)
        }

    legacy_validation_file = APP_DIR / "data" / "keyword_validations.csv"
    if not legacy_validation_file.exists():
        return {}

    saved = pd.read_csv(legacy_validation_file, dtype={"record_identifier": str})
    validations: dict[str, dict[str, list[str]]] = {}
    for row in saved.itertuples(index=False):
        kept_keywords = json.loads(row.kept_keywords)
        if isinstance(kept_keywords, list):
            validations[row.record_identifier] = {
                column: kept_keywords for column in KEYWORD_COLUMNS
            }
        else:
            validations[row.record_identifier] = {
                column: kept_keywords.get(column, []) for column in VALIDATION_COLUMNS
            }
    return validations


def save_validations(validations: dict[str, dict[str, list[str]]]) -> None:
    """Persist the keywords that remain after each review decision as JSON."""
    VALIDATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with VALIDATIONS_FILE.open("w", encoding="utf-8") as validation_file:
        json.dump(validations, validation_file, ensure_ascii=False, indent=2)


def main() -> None:
    st.set_page_config(page_title="Keyword experiment validator", layout="wide")
    st.title("Keyword experiment validator")
    st.caption("Review one record at a time. Remove any keyword that should not be retained.")
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
    render_sidebar_download_button(validations)

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
    is_example_record = record_id == EXAMPLE_RECORD_IDENTIFIER
    previous_column, progress_column, next_column = st.columns([1, 3, 1])
    with previous_column:
        if st.button("← Previous", disabled=position == 0):
            st.session_state.record_position -= 1
            st.rerun()
    with progress_column:
        st.progress((position + 1) / len(records), text=f"Record {position + 1} of {len(records)}")
    with next_column:
        if st.button(
            "Next →", disabled=position == len(records) - 1 or is_example_record
        ):
            st.session_state.record_position += 1
            st.rerun()

    if is_example_record:
        st.info(
            "**Guided example:** remove every illustrative mismatch, add **erosion ratio** "
            "in New keywords from SoilVoc, then select **Save this review**. This example "
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

    st.subheader("Keyword review")
    st.caption("Remove an entry from a list to exclude it from the validated result.")
    saved_keywords = validations.get(record_id, {})
    kept_keywords: dict[str, list[str]] = {}
    for source_column in KEYWORD_COLUMNS:
        available_keywords = split_keywords(record[source_column])
        default_keywords = saved_keywords.get(source_column, available_keywords)
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

    soilvoc_keywords = load_soilvoc_keywords()
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

    if st.button("Save this review", type="primary"):
        example_complete = not is_example_record or example_review_is_complete(kept_keywords)
        if not example_complete:
            st.error(
                "To continue, remove all example mismatches and add 'erosion ratio' "
                "under New keywords from SoilVoc."
            )
        elif not is_example_record:
            validations[record_id] = kept_keywords
            save_validations(validations)
        if example_complete and position < len(records) - 1:
            st.session_state.record_position += 1
            st.rerun()
        if example_complete and is_example_record:
            st.success("Example complete.")
        elif example_complete:
            st.success("Review saved. This is the last record.")


if __name__ == "__main__":
    main()
