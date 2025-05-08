import numpy as np
import pandas as pd
import pathlib as pl
import tomllib as tl

import fra_amtrak.amtk_frame as frm

#1 Read files

# Current working directory
parent_path = pl.Path.cwd()

data_raw_path = parent_path.joinpath("data", "raw")
data_interim_path = parent_path.joinpath("data", "interim")

filepath = parent_path.joinpath("notebook.toml")
with open(filepath, "rb") as file_obj:
    const = tl.load(file_obj)

# Access constants
COLS = const["columns"]

# Check filepaths (this can also be done with the os or glob modules)
filepaths = data_raw_path.glob("*Station%20Performance*.xlsx")
filepaths = [filepath for filepath in filepaths if filepath.is_file()]

# Specify dtypes
dtypes = {
    "Fiscal Year": np.int16,
    "Fiscal Quarter": np.int8,
    "Service Line": "string",
    "Service": "string",
    "Sub Service": "string",
    "Train Number": np.int16,
    "Arrival Station Code": "string",
    "Arrival Station Name": "string",
    "Total Detraining Customers": np.int32,
    "Late Detraining Customers": np.int32,
    # "Avg Min Late (Lt CS)": np.int32,  # Triggers ValueError: invalid literal for int() with base 10: '--'
    # "Avg Min Late (Lt C)": np.int32,  # Triggers ValueError: invalid literal for int() with base 10: '--'
}

# Combine DataFrames
dfs = [pd.read_excel(path, dtype=dtypes) for path in filepaths]
stations = pd.concat(dfs, ignore_index=True)
stations.info()
stations.head()

#2 Drop "Unnamed" column

# Print unique values in Unnamed column
mask = stations.columns[stations.columns.str.contains("^Unnamed")]
print(f"Unnamed columns (n={len(mask)}):")

# Print unique values in unnamed column
unnamed_unique_values = stations[mask].apply(lambda x: x.unique())
print(f"unnamed_unique_values = {unnamed_unique_values}")

# Drop Unnamed column
stations = stations.loc[:, ~stations.columns.str.contains("^Unnamed")]
stations.head()

#3 Transform mixed type columns

# Traverse data frame to detect data types
for column in stations.columns:
    print(f"{column}: ", pd.api.types.infer_dtype(stations[column]))

# Identify the non-numeric values
non_numeric_values = {
    column: frm.find_non_numeric_values(stations, column) for column in stations.columns[-2:]
}

# Return a count of the string values ("--") in the "Avg Min Late" columns.
lt_cs_dashes_count = stations.loc[:, COLS["avg_mm_late_cs"]].str.contains("--").sum()
print(f"Lt CS dashes count = {lt_cs_dashes_count}")

lt_c_dashes_count = stations.loc[:, COLS["avg_mm_late_c"]].str.contains("--").sum()
print(f"Lt C dashes count = {lt_c_dashes_count}")

# Convert dashes to NaN
stations[COLS["avg_mm_late_cs"]] = stations.loc[:, COLS["avg_mm_late_cs"]].replace("--", np.nan).astype(np.float32)
stations[COLS["avg_mm_late_c"]] = stations.loc[:, COLS["avg_mm_late_c"]].replace("--", np.nan).astype(np.float32)
stations.info()

#4 Sort data
stations = stations.sort_values(by=["Fiscal Year", "Fiscal Quarter", "Service Line", "Service", "Sub Service", "Train Number", "Arrival Station Code"],
                     ascending=[False, False, True, True, True, True, True])

#5 Check years, quarters covered
periodsYQ = stations.groupby(["Fiscal Year", "Fiscal Quarter"])["Service"].count().reset_index(name='Rows').sort_values(by='Rows', ascending=False).reset_index(drop=True)
periodsYQSL = stations.groupby(["Fiscal Year", "Fiscal Quarter", "Service Line"])["Service"].count().reset_index(name='Rows').sort_values(by='Rows', ascending=False).reset_index(drop=True)

#6 Persist data
stations.info()
filepath = data_interim_path.joinpath("station_performance_metrics-v1p0.csv")
stations.to_csv(filepath, index=False)