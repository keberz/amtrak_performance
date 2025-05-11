import json
import numpy as np
import pandas as pd
import pathlib as pl
import tomllib as tl

import fra_amtrak.amtk_frame as frm

# Set random seed
rdg = np.random.default_rng(24)

#1 Read files

# Instantiate instances of `pathlib.Path` to represent absolute paths to the `data/interim` and `data/processed`
# directories.
parent_path = pl.Path.cwd()  # current working directory
data_raw_path = parent_path.joinpath("data", "raw")
data_interim_path = parent_path.joinpath("data", "interim")
data_processed_path = parent_path.joinpath("data", "processed")

# Load a companion TOML file containing constants.
filepath = parent_path.joinpath("notebook.toml")
with open(filepath, "rb") as file_obj:
    const = tl.load(file_obj)

# Access constants
COLS = const["columns"]

# Retrieve performance data
filepath = data_interim_path.joinpath("station_performance_metrics-v1p1.csv")
stations = pd.read_csv(filepath)

#2 Add route miles

# Every named train is associated with a route that Amtrak measures in miles.
with open(data_processed_path.joinpath("amtk_sub_services.json"), "r") as file:
    amtk_sub_svcs = json.load(file)

route_miles = [
    {"Route": route["sub service"], "Route Miles": sum([host["miles"] for host in route["hosts"]])}
    for route in amtk_sub_svcs
]

# Create DataFrame
route_miles = pd.DataFrame.from_dict(route_miles, orient="columns")

# Add `route_miles` to the `stations` `DataFrame`. Once the data is combined, move the `route_miles`
# column from the last position to the fifth (`5th`) position in `stations`. Drop any redundant
# columns after reordering the columns.

stations = pd.merge(stations, route_miles, left_on="Service", right_on="Route", how="left")

stations['Route Miles'] = stations['Route Miles'].replace([np.inf, -np.inf], np.nan)
stations['Route Miles'] = stations['Route Miles'].fillna(0).astype(int) # Convert to int with replacement

cols = list(stations.columns) # Get column names
last_col = cols.pop() # Remove the last column
cols.insert(5, last_col) # Insert it at the 5th position
stations = stations[cols] # Reorder the DataFrame
stations = stations.drop(columns=["Route"]) # Drop column

#3 Add location data

# The Bureau of Transportation Statistics (BTS) maintains an Amtrak stations dataset that provides mapping
# (i.e., location) information.

filepath = data_raw_path.joinpath("NTAD_Amtrak_Stations_-3056704789218436106.csv")
ntad_stations = pd.read_csv(filepath)

# Filter out all bus stations and reset the index
ntad_stations = ntad_stations[ntad_stations["StnType"] != "BUS"].reset_index(drop=True)

# Drop columns not required for the analysis
ntad_stations = ntad_stations.drop(columns=["OBJECTID", "StnType", "State", "Name", "StationName", "StationFacilityName", "StationAliases", "DateModif", "x", "y"])

#4 Clean data

# Combined condition to check for empty strings or NaN
mask = (ntad_stations == "") | pd.isna(ntad_stations)
empty_nan_values = ntad_stations.columns[mask.any()]

# Clean strings
ntad_stations = frm.normalize_dataframe_strings(frame=ntad_stations, pattern="\s{2,}")

#5 Manipulate data

# Rename columns
mapper = {
    "StaType": COLS["station_type"],
    "ZipCode": COLS["zip_code"],
    "City": COLS["city"],
    "Address2": COLS["address_02"],
    "Address1": COLS["address_01"],
    "Code": "Code",
    "lon": COLS["lon"],
    "lat": COLS["lat"],
}
ntad_stations.rename(columns=mapper, inplace=True)

# Reorder columns
columns = [
    "Code",
    COLS["station_type"],
    COLS["city"],
    COLS["address_01"],
    COLS["address_02"],
    COLS["zip_code"],
    COLS["lat"],
    COLS["lon"],
]
ntad_stations = ntad_stations.loc[:, columns]

#6 Merge data

# Merge `stations` and `ntad_stations`. Perform a __left join__ to retain all rows in the `stations` `DataFrame`,
# joining on the "Arrival Station Code" column in `stations` and the "Code" column in `ntad_stations`.
stations = pd.merge(stations, ntad_stations, left_on='Arrival Station Code', right_on='Code', how='left')

#7 Check geo coordinates

# Check for missing geo coordinates
missing_coords = stations[["Arrival Station Code", "Arrival Station", "State", "Latitude", "Longitude"]]
missing_coords = missing_coords[missing_coords.isna().any(axis=1)]

# Fix FAL station
values = ("Falmouth", "Muirfield Road at Railroad Crossing", "04105", 43.769600, -70.259500)
mask = stations[COLS["station_code"]] == "FAL"
stations.loc[
    mask,
    [COLS["city"], COLS["address_01"], COLS["zip_code"], COLS["lat"], COLS["lon"]],
] = values

# Fix MSI station
values = ("Michigan City", "100 Washington Street", "46360", 41.721111, -86.905556)
mask = stations[COLS["station_code"]] == "MCI"
stations.loc[
    mask, [COLS["city"], COLS["address_01"], COLS["zip_code"], COLS["lat"], COLS["lon"]]
] = values

#8 Reorder columns

# Indices of interest
state_idx = stations.columns.get_loc(COLS["state"])
total_detrain_idx = stations.columns.get_loc(COLS["total_detrn"])
code_idx = stations.columns.get_loc("Code")

columns_start = stations.columns[:state_idx].tolist()
columns_start.extend([
    "Code",
    COLS["station_type"],
    COLS["city"],
    COLS["address_01"],
    COLS["address_02"],
    COLS["zip_code"],
])
# print(f"columns_start = {columns_start}")

columns_middle = stations.columns[state_idx:total_detrain_idx].tolist()
columns_middle.extend([COLS["lat"], COLS["lon"]])
# print(f"columns_middle = {columns_middle}")

columns_end = stations.columns[total_detrain_idx:code_idx].tolist()
# print(f"columns_end = {columns_end}")

columns = columns_start + columns_middle + columns_end
# print(f"columns = {columns}")

# Reorder DataFrame
stations = stations.loc[:, columns]

#9 Drop column

stations = stations.drop(columns="Code")

#10 Late detraining passengers

# Calculate the ratio of late detraining passengers to total detraining passengers _for each station_
# and assign the results to a new column named "Late to Total Detraining Customers Ratio" (use the
# associated `COLS` constant rather than hard-coding the string name ibnto the code). Round the
# values to the fitfh (`5th`) decimal place.

stations[COLS['late_to_total_detrn_ratio']] = stations.apply(
    lambda row: round(row[COLS['late_detrn']] / row[COLS['total_detrn']], 5) if row[COLS['late_detrn']] else np.nan, axis=1
)

# Reorder columns
cols = list(stations.columns) # Get column names
last_col = cols.pop() # Remove the last column
cols.insert(-1, last_col) # Insert it at the second to last position
stations = stations[cols] # Reorder the DataFrame

#11 Write to file
filepath = data_interim_path.joinpath("station_performance_metrics-v1p2.csv")
stations.to_csv(filepath, index=False)