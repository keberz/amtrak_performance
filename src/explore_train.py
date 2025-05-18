import numpy as np
import pandas as pd
import pathlib as pl
import tomllib as tl

import fra_amtrak.amtk_detrain as detrn
import fra_amtrak.amtk_frame as frm
import fra_amtrak.amtk_network as ntwk
import fra_amtrak.chart_box_preagg as boxp
import fra_amtrak.chart_hist as hst
import fra_amtrak.chart_title as ttl

#1 Read files
parent_path = pl.Path.cwd() # current working directory

filepath = parent_path.joinpath("notebook.toml")
with open(filepath, "rb") as file_obj:
    const = tl.load(file_obj)

# Access constants
AGG = const["agg"]
CHRT_BAR = const["chart"]["bar"]
COLORS = const["colors"]
COLS = const["columns"]
DIRECTION = const["train"]["direction"]
SUB_SVC = const["train"]["sub_service"]
TRN = const["train"]

filepath = parent_path.joinpath("data", "processed", "station_performance_metrics-v1p2.csv")
trains = pd.read_csv(
    filepath, dtype={"Address 02": "str", "ZIP Code": "str"}, low_memory=False
)  # avoid DtypeWarning

#2 Select trains: Northeast Corridor (NEC)

# 2.1 Acela Express (Boston - New York - Philadelphia - Washington, D.C.)
acela_xp = ntwk.by_sub_service(trains, "Acela Express")

# 2.2 Acela Express: on-time performance metrics (entire period)

# Total train arrivals
acela_xp_trn_arrivals = acela_xp.shape[0]

# Detraining totals
acela_xp_detrn = acela_xp[COLS["total_detrn"]].sum()
acela_xp_detrn_late = acela_xp[COLS["late_detrn"]].sum()
acela_xp_detrn_on_time = acela_xp_detrn - acela_xp_detrn_late

# Compute summary statistics
acela_xp_stats = detrn.get_sum_stats(acela_xp, AGG["columns"], AGG["funcs"])

# 2.3 Acela Express trains
acela_xp_trns = acela_xp.drop_duplicates(subset='Train Number', ignore_index=True)
acela_xp_trns = acela_xp_trns[["Service Line", "Service", "Sub Service", "Route Miles", "Train Number"]]
acela_xp_trns.sort_values(by="Train Number", inplace=True, ignore_index=True)

# 2.4 Acela Express: mean late arrival times summary statistics
# Drop missing values
acela_xp_avg_mm_late = acela_xp[COLS["late_detrn_avg_mm_late"]].dropna().reset_index(drop=True)

# Call the custom frm.describe_numeric_column() function again
acela_xp_avg_mm_late_describe = frm.describe_numeric_column(acela_xp_avg_mm_late)

# 2.5 Acela Express: visualize distribution of mean late arrival times

# Convert to DataFrame
acela_xp_avg_mm_late = acela_xp_avg_mm_late.to_frame(name=COLS["avg_mm_late"])

# Get mean and standard deviation
mu = acela_xp_avg_mm_late_describe["center"]["mean"]
sigma = acela_xp_avg_mm_late_describe["spread"]["std"]

# Get max value (for x-axis ticks); pad max value for chart display
max_val = acela_xp_avg_mm_late_describe["position"]["max"]
max_val_ceil = (np.ceil(max_val / 10) * 10).astype(int)

# Create bins
acela_xp_mm_late, bins, num_bins, bin_width = frm.create_bins(
    acela_xp_avg_mm_late, COLS["avg_mm_late"], 10
)

# Bin the data
chrt_data = frm.bin_data(acela_xp_mm_late, COLS["avg_mm_late"], bins)
# chrt_data

# Chart title
title_txt = f"Amtrak {SUB_SVC['ace_xp']} Service Late Detraining Passengers"
title = ttl.format_title(acela_xp_stats, title_txt)

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

# 2.6 Acela Express, Trains 2155 & 2154

# Base columns for routes
rte_cols = [
    COLS["trn"],
    COLS["station_code"],
    COLS["station"],
    COLS["state"],
    COLS["lat"],
    COLS["lon"],
]

# Train 2154 southbound
amtk_2155 = ntwk.by_train_number(trains, 2155)
amtk_2155_rte = ntwk.create_route(amtk_2155, TRN["2154"]["direction"])
amtk_2155_rte_stats = detrn.get_route_sum_stats(
    amtk_2155_rte,
    COLS["station_code"],
    AGG["columns"],
    AGG["funcs"],
    rte_cols,
)
amtk_2155_rte_stats.sort_values(by=[COLS["lat"]], ascending=False, inplace=True)

# Write file
filepath = parent_path.joinpath("data", "student", "stu-amtk_2155_rte_stats.csv")
amtk_2155_rte_stats.to_csv(filepath, index=False)

# Drop missing values
amtk_2155_avg_mm_late = amtk_2155[COLS["late_detrn_avg_mm_late"]].dropna().reset_index(drop=True)

# Describe the column
amtk_2155_avg_mm_late_describe = frm.describe_numeric_column(amtk_2155_avg_mm_late)

# Base columns for average minutes late
cols = [COLS["year"], COLS["quarter"], COLS["late_detrn_avg_mm_late"]]

# Chart data
chrt_data = detrn.get_qtr_avg_min_late(
    amtk_2155_rte, cols, COLS["year_quarter"], [COLORS["amtk_blue"], COLORS["amtk_red"]]
)

# Base columns for aggregation statistics
cols = [COLS["year_quarter"], COLS["late_detrn_avg_mm_late"]]

# Pre-aggregate the data
chrt_data = frm.aggregate_data(chrt_data, cols)

# Create chart title
txt = TRN["2155"]
title_txt = (
    f"Amtrak {txt['name']} Train {txt['number']} Late Detraining Passengers\n"
    f"{txt['route']} ({txt['direction']})"
)
title = ttl.format_title(amtk_2155_rte_stats, title_txt)

# Create and display the vertical boxplot
chart_vertical = boxp.create_boxplot(
    data=chrt_data,
    x_shorthand="Fiscal Year Quarter:N",
    x_title="Period",
    y_shorthand="Late Detraining Customers Avg Min Late:Q",
    y_title="Average Minutes Late",
    box_size=20,
    outlier_shorthand="outliers:Q",
    color_shorthand="Color:N",
    chart_title=title,
    orient=boxp.Orient.VERTICAL,
)
# chart_vertical.display()

# 2154

# YOUR CODE HERE
# Base columns for routes
rte_cols = [
    COLS["trn"],
    COLS["station_code"],
    COLS["station"],
    COLS["state"],
    COLS["lat"],
    COLS["lon"],
]

# Train 2154 southbound
amtk_2154 = ntwk.by_train_number(trains, 2154)
amtk_2154_rte = ntwk.create_route(amtk_2154, TRN["2154"]["direction"])
amtk_2154_rte_stats = detrn.get_route_sum_stats(
    amtk_2154_rte,
    COLS["station_code"],
    AGG["columns"],
    AGG["funcs"],
    rte_cols,
)
amtk_2154_rte_stats.sort_values(by=[COLS["lat"]], ascending=True, inplace=True)

filepath = parent_path.joinpath("data", "student", "stu-amtk_2154_rte_stats.csv")
amtk_2154_rte_stats.to_csv(filepath, index=False)

# Write file
filepath = parent_path.joinpath("data", "student", "stu-amtk_2154_rte_stats.csv")
amtk_2154_rte_stats.to_csv(filepath, index=False)

# Drop missing values
amtk_2154_avg_mm_late = amtk_2154[COLS["late_detrn_avg_mm_late"]].dropna().reset_index(drop=True)

# Describe the column
amtk_2154_avg_mm_late_describe = frm.describe_numeric_column(amtk_2154_avg_mm_late)

# Base columns for average minutes late
cols = [COLS["year"], COLS["quarter"], COLS["late_detrn_avg_mm_late"]]

# Chart data
chrt_data = detrn.get_qtr_avg_min_late(
    amtk_2154_rte, cols, COLS["year_quarter"], [COLORS["amtk_blue"], COLORS["amtk_red"]]
)
# chrt_data

# Base columns for aggregation statistics
cols = [COLS["year_quarter"], COLS["late_detrn_avg_mm_late"]]

# Pre-aggregate the data
chrt_data = frm.aggregate_data(chrt_data, cols)

# Create chart title
txt = TRN["2154"]
title_txt = (
    f"Amtrak {txt['name']} Train {txt['number']} Late Detraining Passengers\n"
    f"{txt['route']} ({txt['direction']})"
)
title = ttl.format_title(amtk_2154_rte_stats, title_txt)

# Create and display the vertical boxplot
chart_vertical = boxp.create_boxplot(
    data=chrt_data,
    x_shorthand="Fiscal Year Quarter:N",
    x_title="Period",
    y_shorthand="Late Detraining Customers Avg Min Late:Q",
    y_title="Average Minutes Late",
    box_size=20,
    outlier_shorthand="outliers:Q",
    color_shorthand="Color:N",
    chart_title=title,
    orient=boxp.Orient.VERTICAL,
)
# chart_vertical.display()

#3 Select trains: State Supported Michigan Service

# 3.1 Pacific Surfliner Service (San Luis Obispo - Santa Barbara - Los Angeles - San Diego)
surf = ntwk.by_sub_service(trains, "Pacific Surfliner")

# 3.2 Pacific Surfliner: on-time performance metrics (entire period)
# Total train arrivals
surf_trn_arrivals = surf.shape[0]

# Detraining totals
surf_detrn = surf[COLS["total_detrn"]].sum()
surf_detrn_late = surf[COLS["late_detrn"]].sum()
surf_detrn_on_time = surf_detrn - surf_detrn_late

# Compute summary statistics
surf_stats = detrn.get_sum_stats(surf, AGG["columns"], AGG["funcs"])

# 3.3 Pacific Surfliner trains
surf_trns = surf.drop_duplicates(subset='Train Number', ignore_index=True)
surf_trns = surf_trns[["Service Line", "Service", "Sub Service", "Route Miles", "Train Number"]]

# 3.4 Pacific Surfliner: mean late arrival times summary statistics
# Drop missing values
surf_avg_mm_late = surf[COLS["late_detrn_avg_mm_late"]].dropna().reset_index(drop=True)

# Call the custom frm.describe_numeric_column() function again
surf_avg_mm_late_describe = frm.describe_numeric_column(surf_avg_mm_late)

# 3.5 Pacific Surfliner: visualize distribution of mean late arrival times
# Convert to DataFrame
surf_avg_mm_late = surf_avg_mm_late.to_frame(name=COLS["avg_mm_late"])

# Get mean and standard deviation
mu = surf_avg_mm_late_describe["center"]["mean"]
sigma = surf_avg_mm_late_describe["spread"]["std"]

# Get max value (for x-axis ticks); pad max value for chart display
max_val = surf_avg_mm_late_describe["position"]["max"]
max_val_ceil = (np.ceil(max_val / 10) * 10).astype(int)

# Create bins
surf_mm_late, bins, num_bins, bin_width = frm.create_bins(surf_avg_mm_late, COLS["avg_mm_late"], 10)

# Bin the data
chrt_data = frm.bin_data(surf_mm_late, COLS["avg_mm_late"], bins)
# chrt_data

# Chart title
title_txt = f"Amtrak {SUB_SVC['surf']} Service Late Detraining Passengers"
title = ttl.format_title(surf_stats, title_txt)

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

# 3.6 Pacific Surfliner Trains 774 & 777

# 774
# YOUR CODE HERE
# Base columns for routes
rte_cols = [
    COLS["trn"],
    COLS["station_code"],
    COLS["station"],
    COLS["state"],
    COLS["lat"],
    COLS["lon"],
]

# Train 774 southbound
amtk_774 = ntwk.by_train_number(trains, 774)
amtk_774_rte = ntwk.create_route(amtk_774, TRN["774"]["direction"])
amtk_774_rte_stats = detrn.get_route_sum_stats(
    amtk_774_rte,
    COLS["station_code"],
    AGG["columns"],
    AGG["funcs"],
    rte_cols,
)
amtk_774_rte_stats.sort_values(by=[COLS["lat"]], ascending=False, inplace=True)

# Write file
# YOUR CODE HERE
filepath = parent_path.joinpath("data", "student", "stu-amtk_774_rte_stats.csv")
amtk_774_rte_stats.to_csv(filepath, index=False)

# Drop missing values
amtk_774_avg_mm_late = amtk_774[COLS["late_detrn_avg_mm_late"]].dropna().reset_index(drop=True)

# Describe the column
amtk_774_avg_mm_late_describe = frm.describe_numeric_column(amtk_774_avg_mm_late)

# Base columns for average minutes late
cols = [COLS["year"], COLS["quarter"], COLS["late_detrn_avg_mm_late"]]

# Chart data
chrt_data = detrn.get_qtr_avg_min_late(
    amtk_774_rte, cols, COLS["year_quarter"], [COLORS["amtk_blue"], COLORS["amtk_red"]]
)

# Base columns for aggregation statistics
cols = [COLS["year_quarter"], COLS["late_detrn_avg_mm_late"]]

# Pre-aggregate the data
chrt_data = frm.aggregate_data(chrt_data, cols)

# Create chart title
txt = TRN["774"]
title_txt = (
    f"Amtrak {txt['name']} Train {txt['number']} Late Detraining Passengers\n"
    f"{txt['route']} ({txt['direction']})"
)
title = ttl.format_title(amtk_774_rte_stats, title_txt)

# Create and display the vertical boxplot
chart_vertical = boxp.create_boxplot(
    data=chrt_data,
    x_shorthand="Fiscal Year Quarter:N",
    x_title="Period",
    y_shorthand="Late Detraining Customers Avg Min Late:Q",
    y_title="Average Minutes Late",
    box_size=20,
    outlier_shorthand="outliers:Q",
    color_shorthand="Color:N",
    chart_title=title,
    orient=boxp.Orient.VERTICAL,
)
# chart_vertical.display()

# 777
# YOUR CODE HERE
# Base columns for routes
rte_cols = [
    COLS["trn"],
    COLS["station_code"],
    COLS["station"],
    COLS["state"],
    COLS["lat"],
    COLS["lon"],
]

# Train 774 southbound
amtk_777 = ntwk.by_train_number(trains, 777)
amtk_777_rte = ntwk.create_route(amtk_777, TRN["777"]["direction"])
amtk_777_rte_stats = detrn.get_route_sum_stats(
    amtk_777_rte,
    COLS["station_code"],
    AGG["columns"],
    AGG["funcs"],
    rte_cols,
)
amtk_777_rte_stats.sort_values(by=[COLS["lat"]], ascending=True, inplace=True)

# Write file
filepath = parent_path.joinpath("data", "student", "stu-amtk_777_rte_stats.csv")
amtk_777_rte_stats.to_csv(filepath, index=False)

# Drop missing values
amtk_777_avg_mm_late = amtk_777[COLS["late_detrn_avg_mm_late"]].dropna().reset_index(drop=True)

# Describe the column
amtk_777_avg_mm_late_describe = frm.describe_numeric_column(amtk_777_avg_mm_late)

# Base columns for average minutes late
cols = [COLS["year"], COLS["quarter"], COLS["late_detrn_avg_mm_late"]]

# Chart data
chrt_data = detrn.get_qtr_avg_min_late(
    amtk_777_rte, cols, COLS["year_quarter"], [COLORS["amtk_blue"], COLORS["amtk_red"]]
)
# chrt_data

# Base columns for aggregation statistics
cols = [COLS["year_quarter"], COLS["late_detrn_avg_mm_late"]]

# Pre-aggregate the data
chrt_data = frm.aggregate_data(chrt_data, cols)

# Create chart title
txt = TRN["777"]
title_txt = (
    f"Amtrak {txt['name']} Train {txt['number']} Late Detraining Passengers\n"
    f"{txt['route']} ({txt['direction']})"
)
title = ttl.format_title(amtk_777_rte_stats, title_txt)

# Create and display the vertical boxplot
chart_vertical = boxp.create_boxplot(
    data=chrt_data,
    x_shorthand="Fiscal Year Quarter:N",
    x_title="Period",
    y_shorthand="Late Detraining Customers Avg Min Late:Q",
    y_title="Average Minutes Late",
    box_size=20,
    outlier_shorthand="outliers:Q",
    color_shorthand="Color:N",
    chart_title=title,
    orient=boxp.Orient.VERTICAL,
)
# chart_vertical.display()

#4 Long-distance trains

# 4.1 City of New Orleans service (Chicago - Memphis - New Orleans)
cno = ntwk.by_sub_service(trains, "City Of New Orleans")

# 4.2 City of New Orleans: on-time performance metrics

# Total train arrivals
cno_trn_arrivals = cno.shape[0]

# Detraining totals
cno_detrn = cno[COLS["total_detrn"]].sum()
cno_detrn_late = cno[COLS["late_detrn"]].sum()
cno_detrn_on_time = cno_detrn - cno_detrn_late

# Compute summary statistics
cno_stats = detrn.get_sum_stats(cno, AGG["columns"], AGG["funcs"])

# 4.3 City of New Orleans trains
cno_trns = cno.drop_duplicates(subset='Train Number', ignore_index=True)
cno_trns = cno_trns[["Service Line", "Service", "Sub Service", "Route Miles", "Train Number"]]

# 4.4 City of New Orleans: mean late arrival times summary statistics

# Drop missing values
cno_avg_mm_late = cno[COLS["late_detrn_avg_mm_late"]].dropna().reset_index(drop=True)

# Call the custom frm.describe_numeric_column() function again
cno_avg_mm_late_describe = frm.describe_numeric_column(cno_avg_mm_late)

# 4.5 City of New Orleans: visualize distribution of mean late arrival times
# Convert to DataFrame
cno_avg_mm_late = cno_avg_mm_late.to_frame(name=COLS["avg_mm_late"])

# Get mean and standard deviation
mu = cno_avg_mm_late_describe["center"]["mean"]
sigma = cno_avg_mm_late_describe["spread"]["std"]

# Get max value (for x-axis ticks); pad max value for chart display
max_val = cno_avg_mm_late_describe["position"]["max"]
max_val_ceil = (np.ceil(max_val / 10) * 10).astype(int)

# Create bins
cno_mm_late, bins, num_bins, bin_width = frm.create_bins(cno_avg_mm_late, COLS["avg_mm_late"], 10)

# Bin the data
chrt_data = frm.bin_data(cno_mm_late, COLS["avg_mm_late"], bins)
# chrt_data

# Chart title
title_txt = f"Amtrak {SUB_SVC['cno']} Service Late Detraining Passengers"
title = ttl.format_title(cno_stats, title_txt)

# Tooltips
tooltip_config = [
    {"shorthand": "bin_center:Q", "title": "Average Minutes Late", "format": None},
    {"shorthand": "count:Q", "title": "Late Arrivals Count", "format": None},
]

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

# 4.6 City of New Orleans, Train 59 and 58

# 59
# Base columns for routes
rte_cols = [COLS["trn"], COLS["station_code"], COLS["station"], COLS["state"], COLS["lat"], COLS["lon"]]

# Train 59 southbound
amtk_59 = ntwk.by_train_number(trains, 59)
amtk_59_rte = ntwk.create_route(amtk_59, "southbound")
amtk_59_rte_stats = detrn.get_route_sum_stats(
    amtk_59_rte, COLS["station_code"], AGG["columns"], AGG["funcs"], rte_cols
)

# Write file
filepath = parent_path.joinpath("data", "student", "stu-amtk_59_rte_stats.csv")
amtk_59_rte_stats.to_csv(filepath, index=False)

# Drop missing values
amtk_59_avg_mm_late = amtk_59[COLS["late_detrn_avg_mm_late"]].dropna().reset_index(drop=True)

# Describe the column
amtk_59_avg_mm_late_describe = frm.describe_numeric_column(amtk_59_avg_mm_late)

# Base columns for chart data
cols = [COLS["year"], COLS["quarter"], COLS["late_detrn_avg_mm_late"]]

# Get the chart data
chrt_data = detrn.get_qtr_avg_min_late(
    amtk_59_rte, cols, COLS["year_quarter"], [COLORS["amtk_blue"], COLORS["amtk_red"]]
)

# Base columns for aggregation statistics
cols = [COLS["year_quarter"], COLS["late_detrn_avg_mm_late"]]

# Pre-aggregate the data
chrt_data = frm.aggregate_data(chrt_data, cols)

# Create chart title
txt = TRN["59"]
title_txt = (
    f"Amtrak {txt['name']} Train {txt['number']} Late Detraining Passengers\n"
    f"{txt['route']} ({txt['direction']})"
)
title = ttl.format_title(amtk_59_rte_stats, title_txt)

# Create and display vertical boxplots
chart_vertical = boxp.create_boxplot(
    data=chrt_data,
    x_shorthand="Fiscal Year Quarter:N",
    x_title="Period",
    y_shorthand="Late Detraining Customers Avg Min Late:Q",
    y_title="Average Minutes Late",
    box_size=20,
    outlier_shorthand="outliers:Q",
    color_shorthand="Color:N",
    chart_title=title,
    orient=boxp.Orient.VERTICAL,
)
# chart_vertical.display()

# 58

# Train 58 northbound
amtk_58 = ntwk.by_train_number(trains, 58)
amtk_58_rte = ntwk.create_route(amtk_58, "northbound")
amtk_58_rte_stats = detrn.get_route_sum_stats(
    amtk_58_rte, COLS["station_code"], AGG["columns"], AGG["funcs"], rte_cols
)

# Write file
filepath = parent_path.joinpath("data", "student", "stu-amtk_58_rte_stats.csv")
amtk_58_rte_stats.to_csv(filepath, index=False)

# Drop missing values
amtk_58_avg_mm_late = amtk_58[COLS["late_detrn_avg_mm_late"]].dropna().reset_index(drop=True)

# Describe the column
amtk_58_avg_mm_late_describe = frm.describe_numeric_column(amtk_58_avg_mm_late)

# Base columns for average minutes late
cols = [COLS["year"], COLS["quarter"], COLS["late_detrn_avg_mm_late"]]

# Chart data
chrt_data = detrn.get_qtr_avg_min_late(
    amtk_58_rte, cols, COLS["year_quarter"], [COLORS["amtk_blue"], COLORS["amtk_red"]]
)

# Base columns for average minutes late
cols = [COLS["year_quarter"], COLS["late_detrn_avg_mm_late"]]

# Pre-aggregate the data
chrt_data = frm.aggregate_data(chrt_data, cols)

# Create chart title
txt = TRN["58"]
title_txt = (
    f"Amtrak {txt['name']} Train {txt['number']} Late Detraining Passengers\n"
    f"{txt['route']} ({txt['direction']})"
)
title = ttl.format_title(amtk_58_rte_stats, title_txt)

# Create and display vertical boxplots
chart_vertical = boxp.create_boxplot(
    data=chrt_data,
    x_shorthand="Fiscal Year Quarter:N",
    x_title="Period",
    y_shorthand="Late Detraining Customers Avg Min Late:Q",
    y_title="Average Minutes Late",
    box_size=20,
    outlier_shorthand="outliers:Q",
    color_shorthand="Color:N",
    chart_title=title,
    orient=boxp.Orient.VERTICAL,
)
# chart_vertical.display()