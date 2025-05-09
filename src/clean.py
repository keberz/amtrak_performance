import json
import numpy as np
import pandas as pd
import pathlib as pl
import re
import tomllib as tl

import fra_amtrak.amtk_frame as frm
import fra_amtrak.amtk_network as ntwk

# Set random seed
rdg = np.random.default_rng(24)

#1 Read files

# Instantiate instances of pathlib.Path to represent absolute paths to the data/interim and data/processed directories.
parent_path = pl.Path.cwd()  # current working directory

data_interim_path = parent_path.joinpath("data", "interim")
data_processed_path = parent_path.joinpath("data", "processed")

# Load a companion TOML file containing constants.
filepath = parent_path.joinpath("notebook.toml")
with open(filepath, "rb") as file_obj:
    const = tl.load(file_obj)

# Access constants
COLS = const["columns"]

# Retrieve performance data (interim)
filepath = data_interim_path.joinpath("station_performance_metrics-v1p0.csv")
stations = pd.read_csv(filepath)

#2 Normalize strings

# Trim each string value of leading/trailing spaces. Also search and remove unnecessary spaces in each string value
# based on the regular expression re.Pattern object. Call the function frm.normalize_dataframe_strings() to
# perform this operation.

stations["Service Line"] = stations["Service Line"].str.strip()
stations["Service"] = stations["Service"].str.strip()
stations["Sub Service"] = stations["Sub Service"].str.strip()
stations["Arrival Station Code"] = stations["Arrival Station Code"].str.strip()
stations["Arrival Station Name"] = stations["Arrival Station Name"].str.strip()
stations = frm.normalize_dataframe_strings(frame=stations, pattern="\s{2,}")

#3 Manipulate data

# Merge "Avg Min Late (Lt CS)" and "Avg Min Late (Lt C)"
mask = stations[COLS["avg_mm_late_c"]].notna()
stations.loc[mask, COLS["avg_mm_late_cs"]] = stations.loc[mask, COLS["avg_mm_late_c"]]
stations[mask].head(3)

# Drop redundant column
stations.drop(columns=["Avg Min Late (Lt C)"], inplace=True)

# Split "Arrival Station Name" string into multiple columns

# The "Arrival Station Name" column is overloaded with location information. The station name, state, and country
# are usually resident in the string.
# Split the column values and unpack the substrings into three new columns named "Arrival Station", "State", and
# "Country". Use the available COLS constants to define the new column names.
temp = stations["Arrival Station Name"].str.split(pat=",", expand=True)
temp = temp.rename(columns={0: "Arrival Station", 1: "State", 2: "Country"})
stations = pd.concat([stations, temp], axis=1)
stations["Arrival Station"] = stations["Arrival Station"].str.strip()
stations["State"] = stations["State"].str.strip()
stations["Country"] = stations["Country"].str.strip()
stations = stations[['Fiscal Year', 'Fiscal Quarter', 'Service Line', 'Service', 'Sub Service',
                     'Train Number', 'Arrival Station Code', 'Arrival Station Name',
                     'Total Detraining Customers', 'Late Detraining Customers',
                     'Avg Min Late (Lt CS)', 'Arrival Station', 'State', 'Country']]

# Update "State" column CA and VT values
stations["State"] = stations["State"].replace(to_replace="CA", value="California")
stations["State"] = stations["State"].replace(to_replace="VT", value="Vermont")

# Update "State" column NaN values
mapper = {"CBN": "New York", "NRG": "California"}
stations[COLS["state"]] = stations[COLS["station_code"]].map(mapper).fillna(stations["State"])

# Update the "Country" column
filepath = data_processed_path.joinpath("states_provinces.json")
with open(filepath, "r") as file_obj:
    states_provinces = json.load(file_obj)

# Define the mapping function
state_to_country = {state: country for country, states in states_provinces.items() for state in states}
def get_country(state):
    return state_to_country.get(state, np.nan)

stations["Country"] = stations["State"].apply(get_country)

# Add region and division columns
filepath = data_processed_path.joinpath("regions_divisions.json")
with open(filepath, "r") as file_obj:
    regions_divisions = json.load(file_obj)

# Assign region to each state, province, and district
stations.loc[:, [COLS["region"], COLS["division"]]] = (
    stations.loc[:, COLS["state"]]
    .apply(lambda x: pd.Series(ntwk.get_region_division(regions_divisions, x)))
    .values
)

# Reorder columns
stations = stations[["Fiscal Year", "Fiscal Quarter", "Service Line", "Service", "Sub Service", "Train Number", "Arrival Station Code", "Arrival Station Name", "Arrival Station",
             "State", "Division", "Region", "Country", "Total Detraining Customers", "Late Detraining Customers", "Avg Min Late (Lt CS)"]]

# Drop "Arrival Station Name" column
stations.drop(columns=["Arrival Station Name"], inplace=True)

# Rename the "Avg Min Late (Lt CS)" column
stations.rename(columns={"Avg Min Late (Lt CS)": "Late Detraining Customers Avg Min Late"}, inplace=True)

#4 Write to file
filepath = data_interim_path.joinpath("station_performance_metrics-v1p1.csv")
stations.to_csv(filepath, index=False)