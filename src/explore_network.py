import numpy as np
import pandas as pd
import pathlib as pl
import scipy.stats as stats
import tomllib as tl

import fra_amtrak.amtk_detrain as detrn
import fra_amtrak.amtk_frame as frm
import fra_amtrak.chart_bar as bar
import fra_amtrak.chart_box_preagg as boxp
import fra_amtrak.chart_hist as hst
import fra_amtrak.chart_title as ttl

#1 Read files
parent_path = pl.Path.cwd()  # current working directory

filepath = parent_path.joinpath("notebook.toml")
with open(filepath, "rb") as file_obj:
    const = tl.load(file_obj)

# Access constants
AGG = const["agg"]
CHRT_BAR = const["chart"]["bar"]
COLORS = const["colors"]
COLS = const["columns"]

# Performance data
filepath = parent_path.joinpath("data", "processed", "station_performance_metrics-v1p2.csv")
network = pd.read_csv(
    filepath, dtype={"Address 02": "str", "ZIP Code": "str"}, low_memory=False
)  # avoid DtypeWarning

#2 The Amtrak network

# Total train arrivals
network_trn_arrivals = network.shape[0]

# Detraining totals
network_detrn = network[COLS["total_detrn"]].sum()
network_detrn_late = network[COLS["late_detrn"]].sum()
network_detrn_on_time = network_detrn - network_detrn_late

# Compute summary statistics
network_stats = detrn.get_sum_stats(network, AGG["columns"], AGG["funcs"])

# Service lines
svc_line_stats = detrn.get_sum_stats_by_group(network, COLS["svc_line"], AGG["columns"], AGG["funcs"])

# Services
serv = network.loc[:, COLS["svc"]].unique()
serv.sort()
svc_stats = detrn.get_sum_stats_by_group(network, COLS["svc"], AGG["columns"], AGG["funcs"])

# Sub services
sub_serv = network.loc[:, COLS["sub_svc"]].unique()
sub_serv.sort()
sub_svc_stats = detrn.get_sum_stats_by_group(network, COLS["sub_svc"], AGG["columns"], AGG["funcs"])

# Stations
stn_count = network.loc[:, COLS["station_code"]].nunique()

# Regions
region_stn_counts = (
    network.groupby(COLS["region"])[COLS["station_code"]]
    .nunique()
    .reset_index()
    .sort_values(by=COLS["region"])
)

# Divisions
div_stn_counts = (
    network.groupby([COLS["region"], COLS["division"]])[COLS["station_code"]]
    .nunique()
    .reset_index()
    .sort_values(by=[COLS["region"], COLS["division"]])
)

#3 On-time performance metrics (entire period)

# Drop missing values
network_avg_mm_late = network[COLS["late_detrn_avg_mm_late"]].dropna().reset_index(drop=True)

# Call the function
network_avg_mm_late_describe = frm.describe_numeric_column(network_avg_mm_late)

# Convert to DataFrame
network_avg_mm_late = network_avg_mm_late.to_frame(name=COLS["avg_mm_late"])

# Get mean and standard deviation
mu = network_avg_mm_late_describe["center"]["mean"]
sigma = network_avg_mm_late_describe["spread"]["std"]

# Get max value (for x-axis ticks); pad max value for chart display
max_val = network_avg_mm_late_describe["position"]["max"].astype(int)
max_val_ceil = (np.ceil(max_val / 10) * 10).astype(int)

# Create bins
network_avg_mm_late, bins, num_bins, bin_width = frm.create_bins(
    network_avg_mm_late, COLS["avg_mm_late"], 15
)

# Bin the data
chrt_data = frm.bin_data(network_avg_mm_late, COLS["avg_mm_late"], bins)

# Chart title
title_txt = "Amtrak Network Late Detraining Passengers"
title = ttl.format_title(network_stats, title_txt)

# Tooltips
tooltip_config = [
    {"shorthand": "bin_center:Q", "title": "Average Minutes Late", "format": None},
    {"shorthand": "count:Q", "title": "Late Arrivals Count", "format": None},
]

# Create and display the histogram
chart = hst.create_histogram(
    frame=chrt_data,
    x_shorthand="bin_center:Q",
    x_title="Average Minutes Late",
    y_shorthand="count:Q",
    y_title="Late Arrivals Count",
    y_stack=False,
    line_shorthand="Avg Min Late:Q",
    mu=mu,
    sigma=sigma,
    num_bins=num_bins,
    bin_width=bin_width,
    x_tick_count_max=max_val_ceil,
    bar_color=COLORS["amtk_blue"],
    mu_color=COLORS["amtk_red"],
    sigma_color=COLORS["anth_gray"],
    tooltip_config=tooltip_config,
    title=title,
    height=300,
    width=680,
)
# chart.display()

#4 On-time performance metrics (by fiscal year and quarter)

# Get quarterly stats
network_qtr_stats = detrn.get_sum_stats_by_group(
    network,
    [COLS["year"], COLS["quarter"]],
    AGG["columns"],
    AGG["funcs"],
    network_trn_arrivals,
    network_detrn,
)

# Save file
filepath = parent_path.joinpath("data", "student", "stu-amtk-network_qtr_stats.csv")
network_qtr_stats.to_csv(filepath, index=False)

#5 Visualize detraining passengers (by fiscal year and quarter)
# Assemble the data for the chart
chrt_data = bar.create_detrain_chart_frame(network_qtr_stats, CHRT_BAR["columns"])

# Create chart title
title_text = f"Amtrak {const['service_lines']['nec']} (NEC) Detraining Passengers"
title = ttl.format_title(network_stats, title_text)

# Grouped bar chart
chart = bar.create_grouped_bar_chart(
    chrt_data,
    "Fiscal Period:N",
    "Passengers:Q",
    "Arrival Status:N",
    CHRT_BAR["xoffset_sort"],
    CHRT_BAR["colors"],
    title,
)
# chart.display()

#6 Visualize distribution of mean late arrival times (by fiscal year and quarter)

cols = [COLS["year"], COLS["quarter"], COLS["late_detrn_avg_mm_late"]]

# Group by fiscal year and quarter and flatten
chrt_data = network.groupby(cols[:2])[cols].apply(lambda x: x).reset_index(drop=True)

# Add 'Fiscal Year Quarter' column
chrt_data["Fiscal Year Quarter"] = chrt_data[["Fiscal Year", "Fiscal Quarter"]].apply(detrn.format_year_quarter, axis=1)

# Drop columns and reorder
chrt_data.drop([COLS["year"], COLS["quarter"]], axis=1, inplace=True)
chrt_data.insert(0, COLS["year_quarter"], chrt_data.pop(COLS["year_quarter"]))

# Add color column
colors = [COLORS["amtk_blue"], COLORS["amtk_red"]]
chrt_data.loc[:, "Color"] = chrt_data[COLS["year_quarter"]].apply(detrn.assign_color, colors=colors)
chrt_data.head()

# Compute aggregation statistics
cols = [COLS["year_quarter"], COLS["late_detrn_avg_mm_late"]]

# Pre-aggregate the data
chrt_data = frm.aggregate_data(chrt_data, cols)

# Create chart title
title_text = "Amtrak Network Late Detraining Passengers"
title = ttl.format_title(network_stats, title_text)

chart_horizontal = boxp.create_boxplot(
    data=chrt_data,
    x_shorthand="Late Detraining Customers Avg Min Late:Q",
    x_title="Average Minutes Late",
    y_shorthand="Fiscal Year Quarter:N",
    y_title="Period",
    box_size=20,
    outlier_shorthand="outliers:Q",
    color_shorthand="Color:N",
    chart_title=title,
    orient=boxp.Orient.HORIZONTAL,
    height=400,
    width=680,
)
# chart_horizontal.display()

#7 Distance traveled and late detraining passengers

lm_data = network[[COLS["route_miles"], COLS["late_detrn_avg_mm_late"]]].reset_index(drop=True)
lm_data_clean = lm_data[["Route Miles", "Late Detraining Customers Avg Min Late"]].dropna()
result = stats.linregress(lm_data_clean["Route Miles"], lm_data_clean["Late Detraining Customers Avg Min Late"])

# Create a new DataFrame named route_mi_intervals comprising a single column named "Route Miles" with values ranging
# from 0 to 2600 in increments of 25.
# Then apply the function detrn.predict_avg_min_late() to each of the "Route Miles" values to generate predicted
# average late times for every twenty-five (25) miles of rail travel up to 2600 miles.
# For the starting zero (0) route mile mark, assign the predicted average late time to 0.0 minutes. Round each
# predicted value to two decimal places. Assign each predicted value to a new column named "Predicted Avg Min Late".
route_mi_intervals = pd.DataFrame({"Route Miles": np.arange(0, 2601, 25)})
route_mi_intervals["Predicted Avg Min Late"] = route_mi_intervals["Route Miles"].apply(lambda x: round(detrn.predict_avg_min_late_by_distance(result, x),2))
route_mi_intervals.loc[0,"Predicted Avg Min Late"] = 0

# Create a DataFrame of predicted late times for named trains to combine with route_mi_intervals. Retrieve each named
# train (e.g. the sub service) and their associate route miles from network and store in two columns named
# "Sub Service" and "Route Miles". Assign the new DataFrame to a variable named trn_route_mi.
#
# Next, generate predictions for each row in trn_route_mi. Round the predictions to the second (2nd) decimal place.
# Assign the predictions to a new column named "Predicted Avg Min Late".
#
# Note that these predicted average late times are relevant only for late detraining passengers who travel the
# entire route.

trn_route_mi = network[["Sub Service", "Route Miles"]].drop_duplicates().reset_index(drop=True)
trn_route_mi["Predicted Avg Min Late"] = trn_route_mi["Route Miles"].apply(lambda x: round(detrn.predict_avg_min_late_by_distance(result, x),2))

# Combine route_mi_intervals and trn_route_mi. Assign the new DataFrame to a variable named lm_predict. Then sort
# the DataFrame rows by the route miles (ascending) and the sub service (descending). Finally, reset the index.

# Columns in play
cols = [COLS["route_miles"], COLS["predict_avg_mm_late"], COLS["sub_svc"]]

# Concatenate DataFrames, sort, and reset the index
lm_predict = pd.concat([route_mi_intervals, trn_route_mi], ignore_index=True)
lm_predict.sort_values(by=[cols[0], cols[-1]], ascending=[True, False], inplace=True)
lm_predict.reset_index(drop=True, inplace=True)

# Write to file
filepath = parent_path.joinpath("data", "student", "stu-amtk-avg_min_late_predict.csv")
lm_predict.to_csv(filepath, index=False)

