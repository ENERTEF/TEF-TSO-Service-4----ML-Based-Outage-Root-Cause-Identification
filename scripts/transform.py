import re
from pathlib import Path
from typing import Final

import pandas as pd

REPORT_ELGO_PARITY: Final[bool] = True

TIMESTAMP_PATTERN: Final[str] = (
    r"\d{2}-\d{2}-\d{2}\s+"
    r"\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,6})?"
)
PRIORITY_PATTERN: Final[str] = r"(?:[A-Z](?:-\d+)?|\*-\d+)"

# AlmText has a 24-character timestamp field and a 4-character priority field.
# Lookaheads validate their contents while the fixed-width matches keep the
# following fields aligned. DOTALL is required for multiline operator comments.
RECORD_PREFIX: Final[str] = (
    rf"^(?=(?P<timestamp>{TIMESTAMP_PATTERN})).{{24}}"
    rf"(?=(?P<prioriteta>{PRIORITY_PATTERN})).{{4}}"
)
POINT_RECORD_RE: Final[re.Pattern[str]] = re.compile(
    RECORD_PREFIX
    + r"(?P<aoj>.{30})"
    + r"(?P<lokacija>.{15})"
    + r"(?P<signal_description>.{26})"
    + r"(?P<sporocilo>.*)$",
    re.DOTALL,
)
NON_POINT_RECORD_RE: Final[re.Pattern[str]] = re.compile(
    RECORD_PREFIX + r"(?P<aoj>.{30})" + r"(?P<source_ref>.{41})" + r"(?P<sporocilo>.*)$",
    re.DOTALL,
)

RECORD_TYPES: Final[dict[str, str]] = {
    "P": "physical_point",
    "A": "application",
    "D": "derived_record",
}

PARSED_COLUMNS: Final[list[str]] = [
    "date_time",
    "Opomba",
    "DevKey",
    "SifraSredstva",
    "point_description",
    "Stanje",
    "Potrditev",
    "prioriteta",
    "record_type",
    "parse_valid",
    "aoj",
    "lokacija",
    "signal_description",
    "naprava_opis",
    "source_ref",
    "sporocilo",
]


def find_project_root(marker: str = "pyproject.toml") -> Path:
    path = Path.cwd().resolve()
    while not (path / marker).exists():
        if path.parent == path:
            raise FileNotFoundError(f"{marker!r} not found above {Path.cwd()}")
        path = path.parent
    return path


def normalize_text(series: pd.Series) -> pd.Series:
    """Collapse whitespace, trim boundaries, and canonicalize empty strings."""
    return series.astype("string").str.replace(r"\s+", " ", regex=True).str.strip().replace("", pd.NA)


def parse_logbook(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the distinct physical-point and non-point AlmText layouts.

    The former parser treated every row as a physical point. Application rows
    therefore split identifiers such as ``s1sms_server`` across ``lokacija`` and
    ``naprava_opis``. The raw ``DevKey`` prefix selects the appropriate regex;
    unknown or structurally invalid rows keep their complete AlmText in
    ``sporocilo`` and are explicitly marked with ``parse_valid=False``.
    """
    alm = df["AlmText"].astype("string")
    devkey = normalize_text(df["DevKey"])
    devkey_prefix = devkey.str.extract(r"^([A-Z]):", expand=False)
    is_point = devkey_prefix.eq("P")

    point_fields = alm.loc[is_point].str.extract(POINT_RECORD_RE)
    non_point_fields = alm.loc[~is_point].str.extract(NON_POINT_RECORD_RE)

    common_columns = ["timestamp", "prioriteta", "aoj", "sporocilo"]
    fields = pd.concat(
        [point_fields[common_columns], non_point_fields[common_columns]],
    ).reindex(df.index)

    parsed_timestamp = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    has_fraction = fields["timestamp"].str.contains(".", regex=False, na=False)
    parsed_timestamp.loc[has_fraction] = pd.to_datetime(
        fields.loc[has_fraction, "timestamp"],
        format="%d-%m-%y %H:%M:%S.%f",
        errors="coerce",
    )
    parsed_timestamp.loc[~has_fraction] = pd.to_datetime(
        fields.loc[~has_fraction, "timestamp"],
        format="%d-%m-%y %H:%M:%S",
        errors="coerce",
    )
    record_type = devkey_prefix.map(RECORD_TYPES).fillna("unknown").astype("string")
    parse_valid = (
        fields["timestamp"].notna()
        & devkey_prefix.isin(RECORD_TYPES)
        & parsed_timestamp.eq(pd.to_datetime(df["date_time"], errors="coerce"))
    )

    embedded_signal_description = normalize_text(point_fields["signal_description"].reindex(df.index))
    raw_point_description = normalize_text(df["PointDesc"])

    parsed = df.assign(
        Opomba=normalize_text(df["Opomba"]),
        DevKey=devkey,
        SifraSredstva=devkey.str.extract(r"\.(\d+)(?:_|$)", expand=False),
        point_description=raw_point_description,
        Stanje=normalize_text(df["Stanje"]),
        Potrditev=normalize_text(df["Potrditev"]),
        prioriteta=normalize_text(fields["prioriteta"]),
        record_type=record_type,
        parse_valid=parse_valid,
        aoj=normalize_text(fields["aoj"]),
        lokacija=normalize_text(point_fields["lokacija"].reindex(df.index)),
        signal_description=embedded_signal_description.where(is_point),
        # Compatibility alias for the engineer-parsed delivery and older notebooks.
        naprava_opis=embedded_signal_description.where(is_point),
        source_ref=normalize_text(non_point_fields["source_ref"].reindex(df.index)),
        sporocilo=normalize_text(fields["sporocilo"]).where(parse_valid, normalize_text(alm)),
    )

    return parsed[PARSED_COLUMNS].copy()


def read_parquet_files(paths: list[Path]) -> pd.DataFrame:
    """Read one or more Parquet files as one row-wise concatenated table."""
    if not paths:
        raise FileNotFoundError("No Parquet input files found")
    return pd.concat((pd.read_parquet(path) for path in sorted(paths)), ignore_index=True)


def normalized_strings(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("")


if __name__ == "__main__":
    PROJECT_ROOT: Final[Path] = find_project_root()
    RAW_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
    EXTRACT_DIR = RAW_DIR / "extracted"
    assert RAW_DIR.exists()
    assert EXTRACT_DIR.exists()

    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    assert PROCESSED_DIR.exists()

    izklopi_files = sorted(EXTRACT_DIR.glob("izklop_*.parquet"))
    izpadi_files = sorted(EXTRACT_DIR.glob("izpad_*.parquet"))
    dcv_devs_files = sorted(RAW_DIR.glob("dcv*.parquet"))
    logbook_raw_files = sorted(RAW_DIR.glob("logbook*raw.parquet"))

    logbook_df = read_parquet_files(logbook_raw_files)
    logbook_parsed_df = parse_logbook(logbook_df)

    if REPORT_ELGO_PARITY:
        # This comparison is diagnostic; ELGO's quick parser is not ground truth.
        logbook_parsed_files = sorted(RAW_DIR.glob("logbook*parsed.parquet"))
        logbook_expected_df = read_parquet_files(logbook_parsed_files)

        cols = [
            "SifraSredstva",
            "prioriteta",
            "aoj",
            "lokacija",
            "naprava_opis",
            "sporocilo",
        ]

        comparison_rows = min(len(logbook_parsed_df), len(logbook_expected_df))
        actual_comparison = logbook_parsed_df.iloc[:comparison_rows].reset_index(drop=True)
        expected_comparison = logbook_expected_df.iloc[:comparison_rows].reset_index(drop=True)

        mismatch_counts = {
            col: int(
                (normalized_strings(actual_comparison[col]) != normalized_strings(expected_comparison[col])).sum()
            )
            for col in cols
        }
        print(
            "ELGO parsed-file comparison "
            f"({len(logbook_parsed_df):,} actual rows, "
            f"{len(logbook_expected_df):,} supplied rows): {mismatch_counts}"
        )

    logbook_parsed_df.to_parquet(
        PROCESSED_DIR / "logbook.parquet",
        index=False,
        compression="zstd",
        compression_level=6,
    )

    izklopi_df = read_parquet_files(izklopi_files)
    izklopi_df.to_parquet(
        PROCESSED_DIR / "izklopi.parquet",
        index=False,
        compression="zstd",
        compression_level=6,
    )

    izpadi_df = read_parquet_files(izpadi_files)
    izpadi_df.to_parquet(
        PROCESSED_DIR / "izpadi.parquet",
        index=False,
        compression="zstd",
        compression_level=6,
    )

    dcv_devs = read_parquet_files(dcv_devs_files)
    dcv_devs.to_parquet(
        PROCESSED_DIR / "dcv_devs.parquet",
        index=False,
        compression="zstd",
        compression_level=6,
    )
