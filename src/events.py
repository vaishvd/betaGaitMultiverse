import pandas as pd

def load_events(events_file):
    """
    Load events.tsv and clean the dataframe.

    Expected columns in this dataset:
    onset | duration | sample | value
    """

    events_df = pd.read_csv(events_file, sep="\t")

    print("Detected columns:", events_df.columns)

    required_cols = {"onset", "duration", "sample", "value"}
    if not required_cols.issubset(events_df.columns):
        raise RuntimeError(
            f"Events file does not contain expected columns.\n"
            f"Found: {events_df.columns}"
        )

    # Convert numeric columns
    events_df["onset"] = pd.to_numeric(events_df["onset"], errors="coerce")
    events_df["duration"] = pd.to_numeric(events_df["duration"], errors="coerce")
    events_df["sample"] = pd.to_numeric(events_df["sample"], errors="coerce")

    # Drop rows with invalid onset
    events_df = events_df.dropna(subset=["onset"])

    # Ensure event labels are strings
    events_df["value"] = events_df["value"].astype(str)

    # Sort events chronologically
    events_df = events_df.sort_values("onset").reset_index(drop=True)

    print("Available event markers:")
    print(events_df["value"].unique())

    return events_df


def find_event(events_df, label):
    """Return onset time for a specific marker."""
    row = events_df[events_df["value"] == label]

    if row.empty:
        raise RuntimeError(f"Event marker '{label}' not found")

    return float(row["onset"].values[0])