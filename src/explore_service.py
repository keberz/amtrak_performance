import numpy as np
import pandas as pd
import pathlib as pl
import tomllib as tl

import fra_amtrak.amtk_detrain as detrn
import fra_amtrak.amtk_frame as frm
import fra_amtrak.amtk_network as ntwk
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
SVC_LINES = const["service_lines"]

filepath = parent_path.joinpath("data", "processed", "station_performance_metrics-v1p2.csv")
network = pd.read_csv(
    filepath, dtype={"Address 02": "str", "ZIP Code": "str"}, low_memory=False
)  # avoid DtypeWarning

#2 Amtrak service lines

svc_line_stats = detrn.get_sum_stats_by_group(
    network, COLS["svc_line"], AGG["columns"], AGG["funcs"]
)

#3 Northeast Corridor (NEC)
nec = ntwk.by_service_line(network, SVC_LINES["nec"])

# Total train arrivals
nec_trn_arrivals = nec.shape[0]

# Detraining totals
nec_detrn = nec[COLS["total_detrn"]].sum()
nec_detrn_late = nec[COLS["late_detrn"]].sum()
nec_detrn_on_time = nec_detrn - nec_detrn_late

# Compute summary statistics
nec_stats = detrn.get_sum_stats(nec, AGG["columns"], AGG["funcs"])

# Drop missing values
nec_avg_mm_late = nec[COLS["late_detrn_avg_mm_late"]].dropna().reset_index(drop=True)

# Call the custom frm.describe_numeric_column() function again
nec_avg_mm_late_describe = frm.describe_numeric_column(nec_avg_mm_late)

# Convert to DataFrame
nec_avg_mm_late = nec_avg_mm_late.to_frame(name=COLS["avg_mm_late"])

# Get mean and standard deviation
mu = nec_avg_mm_late_describe["center"]["mean"]
sigma = nec_avg_mm_late_describe["spread"]["std"]

# Get max value (for x-axis ticks); pad max value for chart display
max_val = nec_avg_mm_late_describe["position"]["max"]
max_val_ceil = (np.ceil(max_val / 10) * 10).astype(int)

# Create bins
nec_mm_late, bins, num_bins, bin_width = frm.create_bins(nec_avg_mm_late, COLS["avg_mm_late"], 15)

# Bin the data
chrt_data = frm.bin_data(nec_mm_late, COLS["avg_mm_late"], bins)

# Chart title
title_txt = f"Amtrak {SVC_LINES['nec']} (NEC) Late Detraining Passengers"
title = ttl.format_title(nec_stats, title_txt)

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
)
# chart.display()

# Get quarterly stats
nec_qtr_stats = detrn.get_sum_stats_by_group(
    nec,
    [COLS["year"], COLS["quarter"]],
    AGG["columns"],
    AGG["funcs"],
    nec_trn_arrivals,
    nec_detrn,
)

# Write to file
filepath = parent_path.joinpath("data", "student", "stu-amtk-nec_qtr_stats.csv")
nec_qtr_stats.to_csv(filepath, index=True)

# Assemble the data for the chart
chrt_data = bar.create_detrain_chart_frame(nec_qtr_stats, CHRT_BAR["columns"])

# Create chart title
title_text = f"Amtrak {SVC_LINES['nec']} (NEC) Detraining Passengers"
title = ttl.format_title(nec_stats, title_text)

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

# Visualize distribution of mean late arrival times (by fiscal year and quarter)

cols = [COLS["year"], COLS["quarter"], COLS["late_detrn_avg_mm_late"]]

# Group by fiscal year and quarter, flatten, and reset index
chrt_data = nec.groupby(cols[:2])[cols].apply(lambda x: x).reset_index(drop=True)

# Add column
chrt_data.loc[:, COLS["year_quarter"]] = chrt_data.apply(detrn.format_year_quarter, axis=1)

# Drop columns and reorder
chrt_data.drop(cols[:2], axis=1, inplace=True)
chrt_data.dropna(inplace=True)
chrt_data.insert(0, COLS["year_quarter"], chrt_data.pop(COLS["year_quarter"]))

# Add alternating colors
colors = [COLORS["amtk_blue"], COLORS["amtk_red"]]
chrt_data.loc[:, "Color"] = chrt_data[COLS["year_quarter"]].apply(detrn.assign_color, colors=colors)
chrt_data.head()

# Compute aggregation statistics
cols = [COLS["year_quarter"], COLS["late_detrn_avg_mm_late"]]

# Pre-aggregate the data
chrt_data = frm.aggregate_data(chrt_data, cols)

# Create chart title
title_text = f"Amtrak {SVC_LINES['nec']} (NEC) Late Detraining Passengers"
title = ttl.format_title(nec_stats, title_text)

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

#4 State Supported services

state = ntwk.by_service_line(network, SVC_LINES["state"])

# Total train arrivals
state_trn_arrivals = state.shape[0]

# Detraining totals
state_detrn = state[COLS["total_detrn"]].sum()
state_detrn_late = state[COLS["late_detrn"]].sum()
state_detrn_on_time = state_detrn - state_detrn_late

# Compute summary statistics
state_stats = detrn.get_sum_stats(state, AGG["columns"], AGG["funcs"])

# Drop missing values
state_avg_mm_late = state[COLS["late_detrn_avg_mm_late"]].dropna().reset_index(drop=True)

# Call the custom frm.describe_numeric_column() function again
state_avg_mm_late_describe = frm.describe_numeric_column(state_avg_mm_late)

# Convert to DataFrame
state_avg_mm_late = state_avg_mm_late.to_frame(name=COLS["avg_mm_late"])

# Get mean and standard deviation
mu = state_avg_mm_late_describe["center"]["mean"]
sigma = state_avg_mm_late_describe["spread"]["std"]

# Get max value (for x-axis ticks); pad max value for chart display
max_val = state_avg_mm_late_describe["position"]["max"]
max_val_ceil = (np.ceil(max_val / 10) * 10).astype(int)

# Create bins
state_min_late, bins, num_bins, bin_width = frm.create_bins(
    state_avg_mm_late, COLS["avg_mm_late"], 15
)

# Bin the data
chrt_data = frm.bin_data(state_min_late, COLS["avg_mm_late"], bins)
# chrt_data

# Chart title
title_txt = f"Amtrak {SVC_LINES['state']} Late Detraining Passengers"
title = ttl.format_title(state_stats, title_txt)

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
    width=680,
)

# Get quarterly stats
state_qtr_stats = detrn.get_sum_stats_by_group(
    state,
    [COLS["year"], COLS["quarter"]],
    AGG["columns"],
    AGG["funcs"],
    state_trn_arrivals,
    state_detrn,
)

# Write to file
filepath = parent_path.joinpath("data", "student", "stu-amtk-state_qtr_stats.csv")
state_qtr_stats.to_csv(filepath, index=True)

# Visualize detraining passengers
# Assemble the data for the chart
chrt_data = bar.create_detrain_chart_frame(state_qtr_stats, CHRT_BAR["columns"])

# Create chart title
title_text = f"Amtrak {SVC_LINES['state']} Detraining Passengers"
title = ttl.format_title(state_stats, title_text)

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

# Distribution of mean late arrival times (by fiscal year and quarter)
cols = [COLS["year"], COLS["quarter"], COLS["late_detrn_avg_mm_late"]]

# Group by fiscal year and quarter, flatten, and reset index
state_avg_mm_late = state.groupby(cols[:2])[cols].apply(lambda x: x).reset_index(drop=True)

# Add column
state_avg_mm_late.loc[:, COLS["year_quarter"]] = state_avg_mm_late.apply(detrn.format_year_quarter, axis=1)

# Drop columns and reorder
state_avg_mm_late.drop(cols[:2], axis=1, inplace=True)
state_avg_mm_late.insert(0, COLS["year_quarter"], state_avg_mm_late.pop(COLS["year_quarter"]))

# Add alternating colors
colors = [COLORS["amtk_blue"], COLORS["amtk_red"]]
state_avg_mm_late.loc[:, "Color"] = state_avg_mm_late[COLS["year_quarter"]].apply(
    detrn.assign_color, colors=colors
)

# Compute aggregation statistics
cols = [COLS["year_quarter"], COLS["late_detrn_avg_mm_late"]]

# Pre-aggregate the data
agg_stats = frm.aggregate_data(state_avg_mm_late, cols)

# Create chart title
title_text = f"Amtrak {SVC_LINES['state']} Late Detraining Passengers"
title = ttl.format_title(nec_stats, title_text)

chart_horizontal = boxp.create_boxplot(
    data=agg_stats,
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

#5 Long Distance services

long_dist = ntwk.by_service_line(network, SVC_LINES["long_dist"])

# Total train arrivals
long_dist_trn_arrivals = long_dist.shape[0]

# Detraining totals
long_dist_detrn = long_dist[COLS["total_detrn"]].sum()
long_dist_detrn_late = long_dist[COLS["late_detrn"]].sum()
long_dist_detrn_on_time = long_dist_detrn - long_dist_detrn_late

print(
    f"Train Arrivals: {long_dist_trn_arrivals}",
    f"Total Detraining Customers: {long_dist_detrn}",
    f"Late Detraining Customers: {long_dist_detrn_late}",
    f"On-Time Detraining Customers: {long_dist_detrn_on_time}",
    sep="\n",
)

# Compute summary statistics
long_dist_stats = detrn.get_sum_stats(long_dist, AGG["columns"], AGG["funcs"])

# Drop missing values
long_dist_avg_min_late = long_dist[COLS["late_detrn_avg_mm_late"]].dropna().reset_index(drop=True)

# Call the custom frm.describe_numeric_column() function again
long_dist_avg_min_late_describe = frm.describe_numeric_column(long_dist_avg_min_late)

# Visualize

# Convert to DataFrame
long_dist_avg_min_late = long_dist_avg_min_late.to_frame(name=COLS["avg_mm_late"])

# Get mean and standard deviation
mu = long_dist_avg_min_late_describe["center"]["mean"]
sigma = long_dist_avg_min_late_describe["spread"]["std"]

# Get max value (for x-axis ticks); pad max value for chart display
max_val = long_dist_avg_min_late_describe["position"]["max"]
max_val_ceil = (np.ceil(max_val / 10) * 10).astype(int)

# Create bins
long_dist_min_late, bins, num_bins, bin_width = frm.create_bins(
    long_dist_avg_min_late, COLS["avg_mm_late"], 10
)

# Bin the data
chrt_data = frm.bin_data(long_dist_min_late, COLS["avg_mm_late"], bins)
# chrt_data

# Chart title
title_txt = f"Amtrak {SVC_LINES['long_dist']} Service Late Detraining Passengers"
title = ttl.format_title(long_dist_stats, title_txt)

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
    width=680,
)
# chart.display()

# Get quarterly stats
long_dist_qtr_stats = detrn.get_sum_stats_by_group(
    long_dist,
    [COLS["year"], COLS["quarter"]],
    AGG["columns"],
    AGG["funcs"],
    long_dist_trn_arrivals,
    long_dist_detrn,
)

# Write to file
filepath = parent_path.joinpath("data", "student", "stu-amtk-long_dist_qtr_stats.csv")
long_dist_qtr_stats.to_csv(filepath, index=True)

# Assemble the data for the chart
chrt_data = bar.create_detrain_chart_frame(long_dist_qtr_stats, CHRT_BAR["columns"])

# Create chart title
title_text = f"Amtrak {SVC_LINES['long_dist']} Detraining Passengers"
title = ttl.format_title(long_dist_stats, title_text)

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

cols = [COLS["year"], COLS["quarter"], COLS["late_detrn_avg_mm_late"]]

# Group by fiscal year and quarter, flatten, and reset index
long_dist_avg_min_late = long_dist.groupby(cols[:2])[cols].apply(lambda x: x).reset_index(drop=True)

# Add column
long_dist_avg_min_late.loc[:, COLS["year_quarter"]] = long_dist_avg_min_late.apply(
    detrn.format_year_quarter, axis=1
)

# Drop columns and reorder
long_dist_avg_min_late.drop(cols[:2], axis=1, inplace=True)
long_dist_avg_min_late.insert(
    0, COLS["year_quarter"], long_dist_avg_min_late.pop(COLS["year_quarter"])
)

# Add alternating colors
colors = [COLORS["amtk_blue"], COLORS["amtk_red"]]
long_dist_avg_min_late.loc[:, "Color"] = long_dist_avg_min_late[COLS["year_quarter"]].apply(
    detrn.assign_color, colors=colors
)

# Compute aggregation statistics
cols = [COLS["year_quarter"], COLS["late_detrn_avg_mm_late"]]

# Pre-aggregate the data
chrt_data = frm.aggregate_data(long_dist_avg_min_late, cols)

# Create chart title
title_text = f"Amtrak {SVC_LINES['long_dist']} Late Detraining Passengers"
title = ttl.format_title(nec_stats, title_text)

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
