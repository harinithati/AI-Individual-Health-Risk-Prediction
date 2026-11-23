"""
Prepare a compact BRFSS 2025 dataset for the AI Health Risk Prediction Streamlit app.

Input:  extracted CDC file LLCP2025.XPT
Output: health_risk_brfss_2025.csv

The output keeps only the 12 project predictors and the target variable.
It retains all rows; it does NOT randomly sample respondents.
"""

from pathlib import Path
import pandas as pd

INPUT_FILE = Path("LLCP2025.XPT")
OUTPUT_FILE = Path("health_risk_brfss_2025.csv")

FEATURES = [
    "_AGEG5YR",
    "SEXVAR",
    "_BMI5CAT",
    "EXERANY2",
    "SMOKE100",
    "DIABETE4",
    "BPHIGH6",
    "GENHLTH",
    "PHYSHLTH",
    "MENTHLTH",
    "EDUCA",
    "INCOME3",
]
TARGET = "CVDCRHD4"
REQUIRED_COLUMNS = FEATURES + [TARGET]

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}. Extract the CDC ZIP and place "
            "LLCP2025.XPT in the same folder as this script."
        )

    print("Reading the original XPT file...")
    df = pd.read_sas(INPUT_FILE, format="xport", encoding="utf-8")

    # BRFSS XPT columns are normally uppercase; normalize names safely.
    df.columns = [str(c).strip().upper() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "These expected project columns were not found in the XPT file: "
            + ", ".join(missing)
            + "\nCheck the BRFSS 2025 codebook and your app's feature names."
        )

    compact = df[REQUIRED_COLUMNS].copy()

    # Convert SAS byte/object values to numeric where possible without
    # changing category codes or deleting rows.
    for col in REQUIRED_COLUMNS:
        compact[col] = pd.to_numeric(compact[col], errors="coerce")

    compact.to_csv(OUTPUT_FILE, index=False)

    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"\nDone: {OUTPUT_FILE.resolve()}")
    print(f"Rows: {len(compact):,}")
    print(f"Columns: {len(compact.columns)}")
    print(f"CSV size: {size_mb:.2f} MB")

    if size_mb >= 200:
        print("WARNING: Output is not below 200 MB.")
    else:
        print("SUCCESS: Output is below the 200 MB limit.")

    print("\nUpload this CSV file to your Streamlit app.")
    print("Note: This keeps all records but only the selected project columns.")
    print("It does not validate clinical meaning or make the model diagnostic.")

if __name__ == "__main__":
    main()
