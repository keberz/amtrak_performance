import json
import numpy as np
import pandas as pd
import pathlib as pl
import tomllib as tl

import fra_amtrak.amtk_detrain as detrn
import fra_amtrak.amtk_frame as frm
import fra_amtrak.amtk_network as ntwk
import fra_amtrak.chart_box_preagg as boxp
import fra_amtrak.chart_hist as hst
import fra_amtrak.chart_hist_layer as hstl
import fra_amtrak.chart_line as lne
import fra_amtrak.chart_title as ttl

# 1 Read files
parent_path = pl.Path.cwd()  # current working directory

filepath = parent_path.joinpath("notebook.toml")
with open(filepath, "rb") as file_obj:
    const = tl.load(file_obj)

# Access constants
AGG = const["agg"]
CHRT_BAR = const["chart"]["bar"]
COLORS = const["colors"]
COLS = const["columns"]
DIRECTION = const["train"]["direction"]
SVC = const["services"]
SUB_SVC = const["train"]["sub_service"]
TRN = const["train"]

filepath = parent_path.joinpath("data", "processed", "amtk_sub_services.json")
with open(filepath, "r") as file:
    amtk_sub_svcs = json.load(file)

filepath = parent_path.joinpath("data", "processed", "amtk_stations.csv")
stations = pd.read_csv(filepath, dtype={"ZIP Code": "str"}, low_memory=False)

filepath = parent_path.joinpath("data", "processed", "station_performance_metrics-v1p2.csv")
trains = pd.read_csv(
    filepath, dtype={"Address 02": "str", "ZIP Code": "str"}, low_memory=False
)  # avoid DtypeWarning

filepath = parent_path.joinpath("data", "student", "stu-amtk-avg_min_late_predict.csv")
predictions = pd.read_csv(filepath, low_memory=False)

#2 State Supported Michigan Service

mich = ntwk.by_service(trains, "Michigan")

#2.1 Michigan service: on-time performance metrics (entire period)
# Total train arrivals
mich_trn_arrivals = mich.shape[0]

# Detraining totals
mich_detrn = mich[COLS["total_detrn"]].sum()
mich_detrn_late = mich[COLS["late_detrn"]].sum()
mich_detrn_on_time = mich_detrn - mich_detrn_late

# Compute summary statistics
mich_stats = detrn.get_sum_stats(mich, AGG["columns"], AGG["funcs"])

#2.2 Michigan service: mean late arrival times
# Drop missing values
mich_avg_mm_late = mich[COLS["late_detrn_avg_mm_late"]].dropna().reset_index(drop=True)

# Call the custom frm.describe_numeric_column() function again
mich_avg_mm_late_describe = frm.describe_numeric_column(mich_avg_mm_late)

#2.3 Michigan service: visualize distribution of mean late arrival times

# Convert to DataFrame
mich_avg_mm_late = mich_avg_mm_late.to_frame(name=COLS["avg_mm_late"])

# Get mean and standard deviation
mich_mu = mich_avg_mm_late_describe["center"]["mean"]
mich_sigma = mich_avg_mm_late_describe["spread"]["std"]

# Get max value (for x-axis ticks); pad max value for chart display
mich_max_val = mich_avg_mm_late_describe["position"]["max"]
mich_max_val_ceil = (np.ceil(mich_max_val / 10) * 10).astype(int)

# Create bins
mich_mm_late, mich_bins, mich_num_bins, mich_bin_width = frm.create_bins(mich_avg_mm_late, COLS["avg_mm_late"], 5)

# Bin the data
chrt_data = frm.bin_data(mich_mm_late, COLS["avg_mm_late"], mich_bins)

# Chart title
title_txt = f"Amtrak {SVC['mich']} Service Late Detraining Passengers"
title = ttl.format_title(mich_stats, title_txt)

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
    mu=mich_mu,
    sigma=mich_sigma,
    num_bins=mich_num_bins,
    bin_width=mich_bin_width,
    x_tick_count_max=mich_max_val_ceil,
    bar_color=COLORS["amtk_blue"],
    mu_color=COLORS["amtk_red"],
    sigma_color=COLORS["anth_gray"],
    tooltip_config=tooltip_config,
    title=title,
)
# chart.display()

#3.0 Michigan sub services: on-time performance metrics (entire period)

mich_sub_svcs_stats = detrn.get_sum_stats_by_group(
    mich,
    [COLS["sub_svc"]],
    AGG["columns"],
    AGG["funcs"],
)

#3.1 Michigan sub services: visualize distribution of mean late arrival times

blwtr = ntwk.by_sub_service(trains, "Blue Water")
prmrq = ntwk.by_sub_service(trains, "Pere Marquette")
wolv = ntwk.by_sub_service(trains, "Wolverine")

# List of sub-services and their mappings
sub_svcs = [
    {"sub_svc": SUB_SVC["blwtr"], "frame": blwtr, "color": COLORS["blue"], "order": 2},
    {"sub_svc": SUB_SVC["prmrq"], "frame": prmrq, "color": COLORS["amtk_red"], "order": 3},
    {"sub_svc": SUB_SVC["wolv"], "frame": wolv, "color": COLORS["amtk_blue"], "order": 1},
]

# Create a three-column DataFrame comprising average late times for each sub-service
mich_sub_svcs = pd.DataFrame({
    sub_svc["sub_svc"]: sub_svc["frame"][COLS["late_detrn_avg_mm_late"]]
    .dropna()
    .reset_index(drop=True)
    for sub_svc in sub_svcs
})

# Melt the DataFrame for charting purposes
chrt_data = pd.melt(
    mich_sub_svcs,
    var_name="Sub Service",
    value_name="Average Minutes Late",
)

# Histograme color and order mappings
hst_colors = {sub_svc["sub_svc"]: sub_svc["color"] for sub_svc in sub_svcs}
hst_order = {sub_svc["sub_svc"]: sub_svc["order"] for sub_svc in sub_svcs}

# Enforce the layering order
chrt_data["order"] = chrt_data[COLS["sub_svc"]].map(hst_order)

# Chart title
title = ttl.format_title(mich_stats, f"Amtrak {SVC['mich']} Service Late Detraining Passengers")

# Tooltip configuration
tooltip_config = [
    {"shorthand": "Sub Service:N", "title": "Sub Service", "format": None},
    {"shorthand": "bin_range:N", "title": "Average Minutes Late (range)", "format": None},
    {"shorthand": "mean_late:Q", "title": "Average Minutes Late (mean)", "format": ".3f"},
    {"shorthand": "count:Q", "title": "Late Arrivals Count", "format": None},
]

chart = hstl.create_layered_histogram(
    frame=chrt_data,
    x_shorthand="bin_start:Q",
    x_title="Average Minutes Late",
    x_tick_count_max=mich_max_val_ceil,
    x2_shorthand="bin_end:Q",
    y_shorthand="count:Q",
    y_title="Late Arrivals Count",
    y_stack=False,
    line_shorthand="Avg Min Late:Q",
    mu=mich_mu,
    sigma=mich_sigma,
    max_bins=mich_num_bins,
    bin_step=5,
    hst_order_shorthand="order:O",
    hst_color_shorthand="Sub Service:N",
    hst_colors=hst_colors,
    mu_color=COLORS["amtk_red"],
    sigma_color=COLORS["anth_gray"],
    tooltip_config=tooltip_config,
    title=title,
)
# chart.display()

#4.0 Michigan Blue Water service

#4.1 Blue Water: on-time performance metrics (entire period)
blwtr_stats = mich_sub_svcs_stats.loc[0,:].squeeze(axis=0)

# Total train arrivals
blwtr_trn_arrivals = blwtr_stats["Train Arrivals"]

# Detraining totals
blwtr_detrn = blwtr_stats[f"{COLS['total_detrn']} sum"]
blwtr_detrn_late = blwtr_stats[f"{COLS['late_detrn']} sum"]
blwtr_detrn_on_time = blwtr_detrn - blwtr_detrn_late

#4.2 Blue Water trains
blwtr_trns = blwtr.drop_duplicates(subset='Train Number', ignore_index=True)
blwtr_trns = blwtr_trns[["Service Line", "Service", "Sub Service", "Route Miles", "Train Number"]]
blwtr_trns.sort_values(by="Train Number", inplace=True, ignore_index=True)

#4.3 Blue Water: mean late arrival times
blwtr_predicted = predictions[predictions["Sub Service"] == "Blue Water"].squeeze(axis=0)
blwtr_avg_mm_late = blwtr[COLS["late_detrn_avg_mm_late"]].dropna().reset_index(drop=True)

# Call the custom frm.describe_numeric_column() function
blwtr_avg_mm_late_describe = frm.describe_numeric_column(blwtr_avg_mm_late)

#4.4 Blue Water: eastbound and westbound routes
blwtr_sub_svc = next(
    (sub_svc for sub_svc in amtk_sub_svcs if sub_svc["sub service"] == SUB_SVC["blwtr"])
)
blwtr_stn_codes = blwtr_sub_svc["station codes"]
blwtr_stns = stations[stations[COLS["station_code"]].isin(blwtr_stn_codes)].reset_index(drop=True)
blwtr_stns.sort_values(by=COLS["lon"], inplace=True)

blwtr_stn_order_eb = blwtr_sub_svc["station order"]["eastbound"]
blwtr_stn_order_wb = blwtr_sub_svc["station order"]["westbound"]

#4.5 Blue Water: eastbound detraining passengers summary statistics

# Base columns for routes
rte_cols = [
    COLS["trn"],
    COLS["station_code"],
    COLS["station"],
    COLS["state"],
    COLS["lat"],
    COLS["lon"],
]

# Train 364 eastbound
amtk_364 = ntwk.by_train_number(trains, 364)
amtk_364_rte = ntwk.create_route(amtk_364, TRN["364"]["direction"], blwtr_stn_order_eb)
amtk_364_rte_stats = detrn.get_route_sum_stats(
    amtk_364_rte, COLS["station_code"], AGG["columns"], AGG["funcs"], rte_cols
)

filepath = parent_path.joinpath("data", "student", "stu-amtk_364_rte_stats.csv")
amtk_364_rte_stats.to_csv(filepath, index=False)

#4.6 Blue Water eastbound mean late arrival times

# Drop missing values
amtk_364_avg_mm_late = amtk_364[COLS["late_detrn_avg_mm_late"]].dropna().reset_index(drop=True)

# Describe the column
amtk_364_avg_mm_late_describe = frm.describe_numeric_column(amtk_364_avg_mm_late)

# Base columns for average minutes late
cols = [COLS["year"], COLS["quarter"], COLS["late_detrn_avg_mm_late"]]

# Chart data
chrt_data = detrn.get_qtr_avg_min_late(
    amtk_364_rte, cols, COLS["year_quarter"], [COLORS["amtk_blue"], COLORS["amtk_red"]]
)

# Base columns for aggregation statistics
cols = [COLS["year_quarter"], COLS["late_detrn_avg_mm_late"]]

# Pre-aggregate the data
chrt_data = frm.aggregate_data(chrt_data, cols)

# Create chart title
txt = TRN["364"]
title_txt = (
    f"Amtrak {txt['name']} Train {txt['number']} Late Detraining Passengers\n"
    f"{txt['route']} ({txt['direction']})"
)
title = ttl.format_title(amtk_364_rte_stats, title_txt)

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

#4.7 Blue Water: visualize eastbound mean late arrival times by station
# Chart title
title_txt = f"Amtrak {SUB_SVC['blwtr']} Service Late Detraining Passengers (2202 Q1 - 2024 Q3)"
title = ttl.format_title(amtk_364_rte_stats, title_txt)

# Arrange stations by direction of travel
x_sort_order = amtk_364_rte_stats.index.tolist()

# Custom line colors
line_colors = {364: COLORS["amtk_blue"]}

# Tooltips
tooltip_config = [
    {"shorthand": f"{COLS['trn']}:N", "title": "Train", "format": None},
    {"shorthand": f"{COLS['station']}:N", "title": "Arrival Station", "format": None},
    {
        "shorthand": f"{COLS['late_detrn_avg_mm_late']} mean",
        "title": "Average Minutes Late",
        "format": None,
    },
]

chart = lne.create_line_chart(
    frame=amtk_364_rte_stats,
    x_shorthand=f"{COLS['station']}:N",
    x_title=f"{COLS['station']}",
    x_sort_order=x_sort_order,
    y_shorthand=f"{COLS['late_detrn_avg_mm_late']} mean:Q",
    y_title="Average Minutes Late",
    y_tick_count_max=75,
    point=True,
    # point={"filled": False, "fill": "white"},
    color_shorthand=f"{COLS['trn']}:N",
    colors=line_colors,
    tooltip_config=tooltip_config,
    title=title,
)
# chart.display()

#4.8 Blue Water: westbound detraining passengers summary statistics

# Base columns for routes
rte_cols = [
    COLS["trn"],
    COLS["station_code"],
    COLS["station"],
    COLS["state"],
    COLS["lat"],
    COLS["lon"],
]

# Train 365 westbound
amtk_365 = ntwk.by_train_number(trains, 365)
amtk_365_rte = ntwk.create_route(amtk_365, TRN["365"]["direction"], blwtr_stn_order_wb)
amtk_365_rte_stats = detrn.get_route_sum_stats(amtk_365_rte, COLS["station_code"], AGG["columns"], AGG["funcs"], rte_cols)

filepath = parent_path.joinpath("data", "student", "stu-amtk_365_rte_stats.csv")
amtk_365_rte_stats.to_csv(filepath, index=False)

#4.9 Blue Water: westbound mean late arrival times

# Drop missing values
amtk_365_avg_mm_late = amtk_365[COLS["late_detrn_avg_mm_late"]].dropna().reset_index(drop=True)

# Describe the column
amtk_365_avg_mm_late_describe = frm.describe_numeric_column(amtk_365_avg_mm_late)

# Base columns for average minutes late
cols = [COLS["year"], COLS["quarter"], COLS["late_detrn_avg_mm_late"]]

# Chart data
chrt_data = detrn.get_qtr_avg_min_late(amtk_365_rte, cols, COLS["year_quarter"], [COLORS["amtk_blue"], COLORS["amtk_red"]])

# Base columns for aggregation statistics
cols = [COLS["year_quarter"], COLS["late_detrn_avg_mm_late"]]

# Pre-aggregate the data
chrt_data = frm.aggregate_data(chrt_data, cols)

# Create chart title
txt = TRN["365"]
title_txt = (
    f"Amtrak {txt['name']} Train {txt['number']} Late Detraining Passengers\n"
    f"{txt['route']} ({txt['direction']})"
)
title = ttl.format_title(amtk_365_rte_stats, title_txt)

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

#4.10 Blue Water: visualize westbound mean late arrival times by station

# Chart title
title_txt = f"Amtrak {SUB_SVC['blwtr']} Service Late Detraining Passengers (2022 Q1 - 2024 Q3)"
title = ttl.format_title(amtk_365_rte_stats, title_txt)

# Arrange stations by direction of travel
x_sort_order = amtk_365_rte_stats.index.tolist()

# Custom line colors
line_colors = {365: COLORS["amtk_blue"]}

# Tooltips
tooltip_config = [
    {"shorthand": f"{COLS['trn']}:N", "title": "Train", "format": None},
    {"shorthand": f"{COLS['station']}:N", "title": "Arrival Station", "format": None},
    {
        "shorthand": f"{COLS['late_detrn_avg_mm_late']} mean",
        "title": "Average Minutes Late",
        "format": None,
    },
]

chart = lne.create_line_chart(
    frame=amtk_365_rte_stats,
    x_shorthand=f"{COLS['station']}:N",
    x_title=f"{COLS['station']}",
    x_sort_order=x_sort_order,
    y_shorthand=f"{COLS['late_detrn_avg_mm_late']} mean:Q",
    y_title="Average Minutes Late",
    y_tick_count_max=75,
    point=True,
    # point={"filled": False, "fill": "white"},
    color_shorthand=f"{COLS['trn']}:N",
    colors=line_colors,
    tooltip_config=tooltip_config,
    title=title,
)
# chart.display()

#5.0 Michigan Pere Marquette service

#5.1 Pere Marquette: on-time performance metrics (entire period)
prmrq_stats = mich_sub_svcs_stats.loc[1,:].squeeze(axis=0)
prmrq_trn_arrivals = prmrq_stats["Train Arrivals"]

# Detraining totals
prmrq_detrn = prmrq_stats[f"{COLS['total_detrn']} sum"]
prmrq_detrn_late = prmrq_stats[f"{COLS['late_detrn']} sum"]
prmrq_detrn_on_time = prmrq_detrn - prmrq_detrn_late

#5.2 Pere Marquette trains
prmrq_trns = prmrq.drop_duplicates(subset='Train Number', ignore_index=True)
prmrq_trns = prmrq_trns[["Service Line", "Service", "Sub Service", "Route Miles", "Train Number"]]
prmrq_trns.sort_values(by="Train Number", inplace=True, ignore_index=True)

#5.3 Pere Marquette: mean late arrival times
prmrq_predicted = predictions[predictions["Sub Service"] == "Pere Marquette"].squeeze(axis=0)

# Drop missing values
prmrq_avg_mm_late = prmrq[COLS["late_detrn_avg_mm_late"]].dropna().reset_index(drop=True)

# Call the custom frm.describe_numeric_column() function
prmrq_avg_mm_late_describe = frm.describe_numeric_column(prmrq_avg_mm_late)

#5.4 Pere Marquette: eastbound and westbound routes
prmrq_sub_svc = next(
    (sub_svc for sub_svc in amtk_sub_svcs if sub_svc["sub service"] == SUB_SVC["prmrq"])
)
prmrq_stn_codes = prmrq_sub_svc["station codes"]
prmrq_stns = stations[stations[COLS["station_code"]].isin(prmrq_stn_codes)].reset_index(drop=True)
prmrq_stns.sort_values(by=COLS["lon"], inplace=True)

prmrq_stn_order_eb = prmrq_sub_svc["station order"]["eastbound"]
prmrq_stn_order_wb = prmrq_sub_svc["station order"]["westbound"]

#5.5 Pere Marquette: eastbound detraining passengers summary statistics

# Base columns for routes
rte_cols = [
    COLS["trn"],
    COLS["station_code"],
    COLS["station"],
    COLS["state"],
    COLS["lat"],
    COLS["lon"],
]

# Train 370 eastbound
amtk_370 = ntwk.by_train_number(trains, 370)
amtk_370_rte = ntwk.create_route(amtk_370, TRN["370"]["direction"], prmrq_stn_order_eb)
amtk_370_rte_stats = detrn.get_route_sum_stats(
    amtk_370_rte, COLS["station_code"], AGG["columns"], AGG["funcs"], rte_cols
)

filepath = parent_path.joinpath("data", "student", "stu-amtk_370_rte_stats.csv")
amtk_370_rte_stats.to_csv(filepath, index=False)

#5.6 Pere Marquette: eastbound mean late arrival times
# Drop missing values
amtk_370_avg_mm_late = amtk_370[COLS["late_detrn_avg_mm_late"]].dropna().reset_index(drop=True)

# Describe the column
amtk_370_avg_mm_late_describe = frm.describe_numeric_column(amtk_370_avg_mm_late)

# Base columns for average minutes late
cols = [COLS["year"], COLS["quarter"], COLS["late_detrn_avg_mm_late"]]

# Chart data
chrt_data = detrn.get_qtr_avg_min_late(
    amtk_370_rte, cols, COLS["year_quarter"], [COLORS["amtk_blue"], COLORS["amtk_red"]]
)

# Base columns for aggregation statistics
cols = [COLS["year_quarter"], COLS["late_detrn_avg_mm_late"]]

# Pre-aggregate the data
chrt_data = frm.aggregate_data(chrt_data, cols)

# Create chart title
txt = TRN["370"]
title_txt = (
    f"Amtrak {txt['name']} Train {txt['number']} Late Detraining Passengers\n"
    f"{txt['route']} ({txt['direction']})"
)
title = ttl.format_title(amtk_370_rte_stats, title_txt)

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

# 5.7 Pere Marquette: visualize eastbound mean late arrival times by station
# Chart title
title_txt = f"Amtrak {SUB_SVC['prmrq']} Service Late Detraining Passengers (2022 Q1 - 2024 Q3)"
title = ttl.format_title(amtk_370_rte_stats, title_txt)

# Arrange stations by direction of travel
x_sort_order = amtk_370_rte_stats.index.tolist()

# Custom line colors
line_colors = {370: COLORS["amtk_blue"]}

# Tooltips
tooltip_config = [
    {"shorthand": f"{COLS['trn']}:N", "title": "Train", "format": None},
    {"shorthand": f"{COLS['station']}:N", "title": "Arrival Station", "format": None},
    {
        "shorthand": f"{COLS['late_detrn_avg_mm_late']} mean",
        "title": "Average Minutes Late",
        "format": None,
    },
]

chart = lne.create_line_chart(
    frame=amtk_370_rte_stats,
    x_shorthand=f"{COLS['station']}:N",
    x_title=f"{COLS['station']}",
    x_sort_order=x_sort_order,
    y_shorthand=f"{COLS['late_detrn_avg_mm_late']} mean:Q",
    y_title="Average Minutes Late",
    y_tick_count_max=75,
    point=True,
    # point={"filled": False, "fill": "white"},
    color_shorthand=f"{COLS['trn']}:N",
    colors=line_colors,
    tooltip_config=tooltip_config,
    title=title,
)
# chart.display()

#5.8 Pere Marquette: westbound detraining passengers summary statistics

# Base columns for routes
rte_cols = [
    COLS["trn"],
    COLS["station_code"],
    COLS["station"],
    COLS["state"],
    COLS["lat"],
    COLS["lon"],
]

# Train 371 westbound
amtk_371 = ntwk.by_train_number(trains, 371)
amtk_371_rte = ntwk.create_route(amtk_371, TRN["371"]["direction"], prmrq_stn_order_wb)
amtk_371_rte_stats = detrn.get_route_sum_stats(
    amtk_371_rte, COLS["station_code"], AGG["columns"], AGG["funcs"], rte_cols
)

filepath = parent_path.joinpath("data", "student", "stu-amtk_371_rte_stats.csv")
amtk_371_rte_stats.to_csv(filepath, index=False)

#5.9 Pere Marquette: westbound mean late arrival times

# Drop missing values
amtk_371_avg_mm_late = amtk_371[COLS["late_detrn_avg_mm_late"]].dropna().reset_index(drop=True)

# Describe the column
amtk_371_avg_mm_late_describe = frm.describe_numeric_column(amtk_371_avg_mm_late)

# Base columns for average minutes late
cols = [COLS["year"], COLS["quarter"], COLS["late_detrn_avg_mm_late"]]

# Chart data
chrt_data = detrn.get_qtr_avg_min_late(
    amtk_371_rte, cols, COLS["year_quarter"], [COLORS["amtk_blue"], COLORS["amtk_red"]]
)

# Base columns for aggregation statistics
cols = [COLS["year_quarter"], COLS["late_detrn_avg_mm_late"]]

# Pre-aggregate the data
chrt_data = frm.aggregate_data(chrt_data, cols)

# Create chart title
txt = TRN["371"]
title_txt = (
    f"Amtrak {txt['name']} Train {txt['number']} Late Detraining Passengers\n"
    f"{txt['route']} ({txt['direction']})"
)
title = ttl.format_title(amtk_371_rte_stats, title_txt)

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

#5.10 Pere Marquette: visualize westbound mean late arrival times by station

# Chart title
title_txt = f"Amtrak {SUB_SVC['prmrq']} Service Late Detraining Passengers (2022 Q1 - 2024 Q3)"
title = ttl.format_title(amtk_371_rte_stats, title_txt)

# Arrange stations by direction of travel
x_sort_order = amtk_371_rte_stats.index.tolist()

# Custom line colors
line_colors = {371: COLORS["amtk_blue"]}

# Tooltips
tooltip_config = [
    {"shorthand": f"{COLS['trn']}:N", "title": "Train", "format": None},
    {"shorthand": f"{COLS['station']}:N", "title": "Arrival Station", "format": None},
    {
        "shorthand": f"{COLS['late_detrn_avg_mm_late']} mean",
        "title": "Average Minutes Late",
        "format": None,
    },
]

chart = lne.create_line_chart(
    frame=amtk_371_rte_stats,
    x_shorthand=f"{COLS['station']}:N",
    x_title=f"{COLS['station']}",
    x_sort_order=x_sort_order,
    y_shorthand=f"{COLS['late_detrn_avg_mm_late']} mean:Q",
    y_title="Average Minutes Late",
    y_tick_count_max=75,
    point=True,
    # point={"filled": False, "fill": "white"},
    color_shorthand=f"{COLS['trn']}:N",
    colors=line_colors,
    tooltip_config=tooltip_config,
    title=title,
)
# chart.display()

#6.0 Michigan Wolverine service

#6.1 Wolverine: on-time performance metrics (entire period)
wolv_stats = mich_sub_svcs_stats.loc[2,:].squeeze(axis=0)

# Total train arrivals
wolv_trn_arrivals = wolv_stats["Train Arrivals"]
# wolv_trn_arrivals

# Detraining totals
wolv_detrn = wolv_stats[f"{COLS['total_detrn']} sum"]
wolv_detrn_late = wolv_stats[f"{COLS['late_detrn']} sum"]
wolv_detrn_on_time = wolv_detrn - wolv_detrn_late

#6.2 Wolverine trains
wolv_trns = wolv.drop_duplicates(subset='Train Number', ignore_index=True)
wolv_trns = wolv_trns[["Service Line", "Service", "Sub Service", "Route Miles", "Train Number"]]
wolv_trns.sort_values(by="Train Number", inplace=True, ignore_index=True)

#6.3 Wolverine: mean late arrival times
wolv_predicted = predictions[predictions["Sub Service"] == "Wolverine"].squeeze(axis=0)

# Drop missing values
wolv_avg_mm_late = wolv[COLS["late_detrn_avg_mm_late"]].dropna().reset_index(drop=True)

# Call the custom frm.describe_numeric_column() function again
wolv_avg_mm_late_describe = frm.describe_numeric_column(wolv_avg_mm_late)

#6.4 Wolverine: eastbound and westbound routes
# Retrieve the sub service from the Amtrak sub services list
wolv_sub_svc = next(
    (sub_svc for sub_svc in amtk_sub_svcs if sub_svc["sub service"] == SUB_SVC["wolv"])
)
wolv_stn_codes = wolv_sub_svc["station codes"]
wolv_stns = stations[stations[COLS["station_code"]].isin(wolv_stn_codes)].reset_index(drop=True)
# WARN: longitude sort does not guarantee correct station order: ROY, TRM, PNT last/first 3 stops
wolv_stns.sort_values(by=COLS["lon"], inplace=True)

wolv_stn_order_eb = wolv_sub_svc["station order"]["eastbound"]
wolv_stn_order_wb = wolv_sub_svc["station order"]["westbound"]

#6.5 Wolverine: eastbound detraining passengers summary statistics
wolv_eb = wolv[wolv["Train Number"].isin([350, 352, 354])]

wolv_eb_stats = detrn.get_sum_stats_by_group(
    wolv_eb,
    COLS["sub_svc"],
    AGG["columns"],
    AGG["funcs"],
)
wolv_eb_stats.drop(columns="Sub Service", inplace=True)

rte_cols = [
    COLS["trn"],
    COLS["station_code"],
    COLS["station"],
    COLS["state"],
    COLS["lat"],
    COLS["lon"],
]

# Train 350 eastbound
amtk_350 = ntwk.by_train_number(trains, 350)
amtk_350_rte = ntwk.create_route(amtk_350, TRN["350"]["direction"], wolv_stn_order_eb)
amtk_350_rte_stats = detrn.get_route_sum_stats(
    amtk_350_rte, COLS["station_code"], AGG["columns"], AGG["funcs"], rte_cols
)

filepath = parent_path.joinpath("data", "student", "stu-amtk_350_rte_stats.csv")
amtk_350_rte_stats.to_csv(filepath, index=False)

# Train 352 eastbound
amtk_352 = ntwk.by_train_number(wolv, 352)
amtk_352_rte = ntwk.create_route(amtk_352, TRN["352"]["direction"], wolv_stn_order_eb)
amtk_352_rte_stats = detrn.get_route_sum_stats(
    amtk_352_rte, COLS["station_code"], AGG["columns"], AGG["funcs"], rte_cols
)

filepath = parent_path.joinpath("data", "student", "stu-amtk_352_rte_stats.csv")
amtk_352_rte_stats.to_csv(filepath, index=False)

# Train 354 eastbound
amtk_354 = ntwk.by_train_number(wolv, 354)
amtk_354_rte = ntwk.create_route(amtk_354, TRN["354"]["direction"], wolv_stn_order_eb)
amtk_354_rte_stats = detrn.get_route_sum_stats(
    amtk_354_rte, COLS["station_code"], AGG["columns"], AGG["funcs"], rte_cols
)

filepath = parent_path.joinpath("data", "student", "stu-amtk_354_rte_stats.csv")
amtk_354_rte_stats.to_csv(filepath, index=False)

#6.6 Wolverine: eastbound mean late arrival times

# 350
# Drop missing values
amtk_350_avg_mm_late = amtk_350[COLS["late_detrn_avg_mm_late"]].dropna().reset_index(drop=True)

# Describe the column
amtk_350_avg_mm_late_describe = frm.describe_numeric_column(amtk_350_avg_mm_late)

#352
# Drop missing values
amtk_352_avg_mm_late = amtk_352[COLS["late_detrn_avg_mm_late"]].dropna().reset_index(drop=True)

# Describe the column
amtk_352_avg_mm_late_describe = frm.describe_numeric_column(amtk_352_avg_mm_late)

#354
# Drop missing values
amtk_354_avg_mm_late = amtk_354[COLS["late_detrn_avg_mm_late"]].dropna().reset_index(drop=True)

# Describe the column
amtk_354_avg_mm_late_describe = frm.describe_numeric_column(amtk_354_avg_mm_late)

# Base columns for average minutes late
cols = [COLS["year"], COLS["quarter"], COLS["late_detrn_avg_mm_late"]]

# Data for boxplots
wolv_eb_trns = [
    {"number": 350, "route": amtk_350_rte, "stats": amtk_350_rte_stats},
    {"number": 352, "route": amtk_352_rte, "stats": amtk_352_rte_stats},
    {"number": 354, "route": amtk_354_rte, "stats": amtk_354_rte_stats},
]

# Assemble charts
for trn in wolv_eb_trns:
    chrt_data = detrn.get_qtr_avg_min_late(
        trn["route"], cols, COLS["year_quarter"], [COLORS["amtk_blue"], COLORS["amtk_red"]]
    )

    # Pre-aggregate the data
    chrt_data = frm.aggregate_data(
        chrt_data, [COLS["year_quarter"], COLS["late_detrn_avg_mm_late"]]
    )

    # Create chart title
    txt = TRN[str(trn["number"])]
    title_txt = (
        f"Amtrak {txt['name']} Train {txt['number']} Late Detraining Passengers\n"
        f"{txt['route']} ({txt['direction']})"
    )
    title = ttl.format_title(trn["stats"], title_txt)

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

#6.7 Wolverine: visualize eastbound mean late arrival times by station

wolv_stns = wolv_stns.reindex(columns=amtk_350_rte_stats.columns).reset_index(drop=True)

amtk_350_chrt_data = ntwk.add_stations_to_route(
    amtk_350_rte_stats.copy(),
    wolv_stns[wolv_stns[COLS["station_code"]] == "ALI"],
    wolv_stn_order_eb,
)

amtk_352_chrt_data = ntwk.add_stations_to_route(
    amtk_352_rte_stats.copy(),
    wolv_stns[wolv_stns[COLS["station_code"]].isin(("ALI", "DOA", "HMI", "MCI"))],
    wolv_stn_order_eb,
)

amtk_354_chrt_data = ntwk.add_stations_to_route(
    amtk_354_rte_stats.copy(),
    wolv_stns[wolv_stns[COLS["station_code"]].isin(("DOA", "HMI"))],
    wolv_stn_order_eb,
)

chrt_data = pd.concat([amtk_350_chrt_data, amtk_352_chrt_data, amtk_354_chrt_data], ignore_index=True)

# Chart title
title_txt = f"Amtrak {SUB_SVC['wolv']} Service Late Detraining Passengers"
title = ttl.format_title(wolv_eb_stats, title_txt)

# Arrange stations by direction of travel
x_sort_order = chrt_data.index.tolist()

# Custom line colors
line_colors = {350: COLORS["amtk_blue"], 352: COLORS["amtk_red"], 354: COLORS["blue"]}

# Tooltips
tooltip_config = [
    {"shorthand": f"{COLS['trn']}:N", "title": "Train", "format": None},
    {"shorthand": f"{COLS['station']}:N", "title": "Arrival Station", "format": None},
    {
        "shorthand": f"{COLS['late_detrn_avg_mm_late']} mean",
        "title": "Average Minutes Late",
        "format": None,
    },
]

chart = lne.create_line_chart_interp(
    frame=chrt_data,
    x_shorthand=f"{COLS['station']}:N",
    x_title=f"{COLS['station']}",
    x_sort_order=x_sort_order,
    y_shorthand=f"{COLS['late_detrn_avg_mm_late']} mean:Q",
    y_title="Average Minutes Late",
    y_tick_count_max=85,
    point=True,
    # point={"filled": False, "fill": "white"},
    color_shorthand=f"{COLS['trn']}:N",
    colors=line_colors,
    tooltip_config=tooltip_config,
    title=title,
)
# chart.display()

#6.8 Wolverine: westbound detraining passengers summary statistics

wolv_wb = wolv[wolv["Train Number"].isin([351, 353, 355])]

wolv_wb_stats = detrn.get_sum_stats_by_group(
    wolv_wb,
    COLS["sub_svc"],
    AGG["columns"],
    AGG["funcs"],
)
wolv_wb_stats.drop(columns="Sub Service", inplace=True)

# Train 351 westbound
amtk_351 = ntwk.by_train_number(wolv, 351)
amtk_351_rte = ntwk.create_route(amtk_351, TRN["351"]["direction"], wolv_stn_order_wb)
amtk_351_rte_stats = detrn.get_route_sum_stats(
    amtk_351_rte, COLS["station_code"], AGG["columns"], AGG["funcs"], rte_cols
)

filepath = parent_path.joinpath("data", "student", "stu-amtk_351_rte_stats.csv")
amtk_351_rte_stats.to_csv(filepath, index=False)

# Train 353 westbound
amtk_353 = ntwk.by_train_number(wolv, 353)
amtk_353_rte = ntwk.create_route(amtk_353, TRN["353"]["direction"], wolv_stn_order_wb)
amtk_353_rte_stats = detrn.get_route_sum_stats(
    amtk_353_rte, COLS["station_code"], AGG["columns"], AGG["funcs"], rte_cols
)

filepath = parent_path.joinpath("data", "student", "stu-amtk_353_rte_stats.csv")
amtk_353_rte_stats.to_csv(filepath, index=False)

# Train 355 westbound
amtk_355 = ntwk.by_train_number(wolv, 355)
amtk_355_rte = ntwk.create_route(amtk_355, TRN["355"]["direction"], wolv_stn_order_wb)
amtk_355_rte_stats = detrn.get_route_sum_stats(
    amtk_355_rte, COLS["station_code"], AGG["columns"], AGG["funcs"], rte_cols
)

filepath = parent_path.joinpath("data", "student", "stu-amtk_355_rte_stats.csv")
amtk_355_rte_stats.to_csv(filepath, index=False)

#351
# Drop missing values
amtk_351_avg_mm_late = amtk_351[COLS["late_detrn_avg_mm_late"]].dropna().reset_index(drop=True)

# Describe the column
amtk_351_avg_mm_late_describe = frm.describe_numeric_column(amtk_351_avg_mm_late)

#353
# Drop missing values
amtk_353_avg_mm_late = amtk_353[COLS["late_detrn_avg_mm_late"]].dropna().reset_index(drop=True)

# Describe the column
amtk_353_avg_mm_late_describe = frm.describe_numeric_column(amtk_353_avg_mm_late)

#355
# Drop missing values
amtk_355_avg_mm_late = amtk_355[COLS["late_detrn_avg_mm_late"]].dropna().reset_index(drop=True)

# Describe the column
amtk_355_avg_mm_late_describe = frm.describe_numeric_column(amtk_355_avg_mm_late)

# Base columns for average minutes late
cols = [COLS["year"], COLS["quarter"], COLS["late_detrn_avg_mm_late"]]

# Data for boxplots
wolv_wb_trns = [
    {"number": 351, "route": amtk_351_rte, "stats": amtk_351_rte_stats},
    {"number": 353, "route": amtk_353_rte, "stats": amtk_353_rte_stats},
    {"number": 355, "route": amtk_355_rte, "stats": amtk_355_rte_stats},
]

# Assemble charts
for trn in wolv_wb_trns:
    chrt_data = detrn.get_qtr_avg_min_late(
        trn["route"], cols, COLS["year_quarter"], [COLORS["amtk_blue"], COLORS["amtk_red"]]
    )

    # Pre-aggregate the data
    chrt_data = frm.aggregate_data(
        chrt_data, [COLS["year_quarter"], COLS["late_detrn_avg_mm_late"]]
    )

    # Create chart title
    txt = TRN[str(trn["number"])]
    title_txt = (
        f"Amtrak {txt['name']} Train {txt['number']} Late Detraining Passengers\n"
        f"{txt['route']} ({txt['direction']})"
    )
    title = ttl.format_title(trn["stats"], title_txt)

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

#6.10 Wolverine: visualize westbound mean late arrival times by station
amtk_351_chrt_data = ntwk.add_stations_to_route(
    amtk_351_rte_stats.copy(),
    wolv_stns[wolv_stns[COLS["station_code"]].isin(("DOA", "HMI", "MCI", "NBU", "NLS"))],
    wolv_stn_order_wb,
)

amtk_353_chrt_data = ntwk.add_stations_to_route(
    amtk_353_rte_stats.copy(),
    wolv_stns[wolv_stns[COLS["station_code"]].isin(("ALI", "DOA", "MCI"))],
    wolv_stn_order_wb,
)

amtk_355_chrt_data = ntwk.add_stations_to_route(
    amtk_355_rte_stats.copy(),
    wolv_stns[wolv_stns[COLS["station_code"]] == "ALI"],
    wolv_stn_order_wb,
)

chrt_data = pd.concat([amtk_351_chrt_data, amtk_353_chrt_data, amtk_355_chrt_data], ignore_index=True)

# Chart title
title_txt = f"Amtrak {SUB_SVC['wolv']} Service Late Detraining Passengers"
title = ttl.format_title(wolv_wb_stats, title_txt)

# Arrange stations by direction of travel
x_sort_order = chrt_data.index.tolist()

# Custom line colors
line_colors = {351: COLORS["amtk_blue"], 353: COLORS["amtk_red"], 355: COLORS["blue"]}

# Tooltips
tooltip_config = [
    {"shorthand": f"{COLS['trn']}:N", "title": "Train", "format": None},
    {"shorthand": f"{COLS['station']}:N", "title": "Arrival Station", "format": None},
    {
        "shorthand": f"{COLS['late_detrn_avg_mm_late']} mean",
        "title": "Average Minutes Late",
        "format": None,
    },
]

chart = lne.create_line_chart_interp(
    frame=chrt_data,
    x_shorthand=f"{COLS['station']}:N",
    x_title=f"{COLS['station']}",
    x_sort_order=x_sort_order,
    y_shorthand=f"{COLS['late_detrn_avg_mm_late']} mean:Q",
    y_title="Average Minutes Late",
    y_tick_count_max=85,
    point=True,
    # point={"filled": False, "fill": "white"},
    color_shorthand=f"{COLS['trn']}:N",
    colors=line_colors,
    tooltip_config=tooltip_config,
    title=title,
)
# chart.display()