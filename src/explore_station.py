import numpy as np
import pandas as pd
import pathlib as pl
import tomllib as tl

import fra_amtrak.amtk_detrain as detrn
import fra_amtrak.amtk_frame as frm
import fra_amtrak.amtk_network as ntwk
import fra_amtrak.chart_bar as vis_bar
import fra_amtrak.chart_box as box
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
CHRT_BOX = const["chart"]["box"]
COLORS = const["colors"]
COLS = const["columns"]
STNS = const["stations"]

filepath = parent_path.joinpath("data", "processed", "station_performance_metrics-v1p2.csv")
stations = pd.read_csv(
    filepath, dtype={"Address 02": "str", "ZIP Code": "str"}, low_memory=False
)  # avoid DtypeWarning

#2 Passenger arrivals

# Columns of interest (for display output only)
cols = [
    COLS["station_code"],
    COLS["station"],
    COLS["city"],
    COLS["state"],
    COLS["division"],
    COLS["region"],
    COLS["total_detrn"],
]

# Top-10 stations
top_n_stns = ntwk.get_n_busiest_stations(stations, 10)[cols]

# Top 10 stations (2023 Q1-Q2)
top_n_stns_filtered = ntwk.get_n_busiest_stations(stations, 10, None, 2023, 1, 2)[cols]

# Top 3 stations (by region, entire period)
region_top_n_stns = ntwk.get_n_busiest_stations(stations, 3, COLS["region"])[cols]

# Top 3 stations (by division, entire period)
div_top_n_stns = ntwk.get_n_busiest_stations(stations, 3, COLS["division"])[cols]

# Top 3 stations (by state)
state_top_n_stns = ntwk.get_n_busiest_stations(stations, 3, COLS["state"])[cols]

#3 Select Station metrics - NYP

# All fiscal years and quarters
nyp = ntwk.by_station(stations, "NYP")

# Train arrivals (total)
nyp_trn_arrivals = nyp.shape[0]

# Detraining totals
nyp_detrn = nyp[COLS["total_detrn"]].sum()
nyp_detrn_late = nyp[COLS["late_detrn"]].sum()
nyp_detrn_on_time = nyp_detrn - nyp_detrn_late

nyp_stats = detrn.get_sum_stats(nyp, AGG["columns"], AGG["funcs"])

# Drop missing values
nyp_avg_mm_late = nyp[COLS["late_detrn_avg_mm_late"]].dropna().reset_index(drop=True)

# Call the custom frm.describe_numeric_column() function again
nyp_avg_mm_late_describe = frm.describe_numeric_column(nyp_avg_mm_late)

# Convert to DataFrame
nyp_avg_mm_late = nyp_avg_mm_late.to_frame(name=COLS["avg_mm_late"])

# Get mean and standard deviation
mu = nyp_avg_mm_late_describe["center"]["mean"]
sigma = nyp_avg_mm_late_describe["spread"]["std"]

# Get max value (for x-axis ticks); pad max value for chart display
max_val = nyp_avg_mm_late_describe["position"]["max"]
max_val_ceil = (np.ceil(max_val / 10) * 10).astype(int)

# Create bins
nyp_min_late, bins, num_bins, bin_width = frm.create_bins(nyp_avg_mm_late, COLS["avg_mm_late"], 15)

# Bin the data
chrt_data = frm.bin_data(nyp_min_late, COLS["avg_mm_late"], bins)

# Chart title
title_txt = f"Late Detraining Passengers: {STNS['nyp']}"
title = ttl.format_title(nyp_stats, title_txt)

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

# Get quarterly stats
nyp_qtr_stats = detrn.get_sum_stats_by_group(
    nyp,
    [COLS["year"], COLS["quarter"]],
    AGG["columns"],
    AGG["funcs"],
    nyp_trn_arrivals,
    nyp_detrn,
)
nyp_qtr_stats.sort_values(by=[COLS["year"], COLS["quarter"]], ascending=[True, True])

# Write to file
filepath = parent_path.joinpath("data", "student", "stu-amtk-nyp_qtr_stats.csv")
nyp_qtr_stats.to_csv(filepath, index=True)

# Assemble the data for the chart
chrt_data = vis_bar.create_detrain_chart_frame(nyp_qtr_stats, CHRT_BAR["columns"])

# Get station code, station name, city, and state to use in the chart title
text = frm.drop_dups_and_squeeze(
    nyp, [COLS["station_code"], COLS["station"], COLS["city"], COLS["state"]]
)

# Chart title
title_txt = (
    f"Detraining Passengers: {text['Arrival Station']} ({text['Arrival Station Code']}), "
    f"{text['City']}, {text['State']}"
)
title = ttl.format_title(nyp_stats, title_txt)

# Create and display grouped bar chart
chart = vis_bar.create_grouped_bar_chart(
    chrt_data,
    "Fiscal Period:N",
    "Passengers:Q",
    "Arrival Status:N",
    CHRT_BAR["xoffset_sort"],
    CHRT_BAR["colors"],
    title,
)
# chart.display()

nyp_svc_trns = nyp.groupby(COLS["svc_line"]).size().reset_index()  # Includes rows with NaN
nyp_svc_trns.columns = [COLS["svc_line"], COLS["trn_arrivals"]]
nyp_svc_trns.sort_values(by=COLS["trn_arrivals"], ascending=False, inplace=True)
nyp_svc_trns.reset_index(drop=True, inplace=True)

# Add train arrival ratios (year_qtr/total)
nyp_svc_trns.loc[:, COLS["trn_arrival_ratio"]] = (
    nyp_svc_trns[COLS["trn_arrivals"]] / nyp_trn_arrivals
)

# Get summary stats by COLS["svc_line"]
nyp_svc_line_stats = detrn.get_sum_stats_by_group(
    nyp, COLS["svc_line"], AGG["columns"], AGG["funcs"]
)

# Merge train arrivals by service line
nyp_svc_line_stats = nyp_svc_line_stats.merge(nyp_svc_trns, on=COLS["svc_line"], how="inner")

# Move train arrival columns
cols = nyp_svc_line_stats.columns.tolist()
cols = [cols[0]] + cols[-2:] + cols[1:-2]
nyp_svc_line_stats = nyp_svc_line_stats[cols]

# Add service line detraining ratios
nyp_svc_line_stats.loc[:, "Service Line Detraining Ratio"] = (
    nyp_svc_line_stats["Total Detraining Customers sum"] / nyp_detrn
)

# Move service line detraining ratio column
nyp_svc_line_stats.insert(
    3, "Service Line Detraining Ratio", nyp_svc_line_stats.pop("Service Line Detraining Ratio")
)

# Sort by passengers detrained (descending order)
nyp_svc_line_stats.sort_values(by="Total Detraining Customers sum", ascending=False, inplace=True)

# Reset index
nyp_svc_line_stats.reset_index(drop=True, inplace=True)

# Visualize distribution of mean late arrival times
nyp_svc_lines = nyp.groupby(COLS["svc_line"])[[COLS["svc_line"], COLS["late_detrn_avg_mm_late"]]]
chrt_data = nyp_svc_lines.apply(lambda x: x).reset_index(drop=True)  # Flatten for Altair

title_txt = (
    f"Detraining Passengers: {text['Arrival Station']} ({text['Arrival Station Code']}), "
    f"{text['City']}, {text['State']}"
)
title = ttl.format_title(nyp_stats, title_txt)

# Create and display the box plots
chart = box.create_box_plot(
    chrt_data,
    "Late Detraining Customers Avg Min Late:Q",
    "Average Minutes Late",
    "Service Line:N",
    COLS["svc_line"],
    CHRT_BOX["y_axis"]["sort"],
    CHRT_BOX["colors"],
    title,
    CHRT_BOX["padding"],
)

# chart.display()

#3 Select Station metrics - Chicago Union Station (CHI), Chicago, IL

chi = ntwk.by_station(stations, "CHI")

# Train arrivals (total)
chi_trn_arrivals = chi.shape[0]

# Detraining totals
chi_detrn = chi[COLS["total_detrn"]].sum()
chi_detrn_late = chi[COLS["late_detrn"]].sum()
chi_detrn_on_time = chi_detrn - chi_detrn_late

chi_stats = detrn.get_sum_stats(chi, AGG["columns"], AGG["funcs"])

# Drop missing values
chi_avg_mm_late = chi[COLS["late_detrn_avg_mm_late"]].dropna().reset_index(drop=True)

# Call the custom frm.describe_numeric_column() function again
chi_avg_mm_late_describe = frm.describe_numeric_column(chi_avg_mm_late)
chi_avg_mm_late_describe

# Convert to DataFrame
chi_avg_mm_late = chi_avg_mm_late.to_frame(name=COLS["avg_mm_late"])

# Get mean and standard deviation
mu = chi_avg_mm_late_describe["center"]["mean"]
sigma = chi_avg_mm_late_describe["spread"]["std"]

# Get max value (for x-axis ticks); pad max value for chart display
max_val = chi_avg_mm_late_describe["position"]["max"]
max_val_ceil = (np.ceil(max_val / 10) * 10).astype(int)

# Create bins
chi_min_late, bins, num_bins, bin_width = frm.create_bins(chi_avg_mm_late, COLS["avg_mm_late"], 10)

# Bin the data
chrt_data = frm.bin_data(chi_min_late, COLS["avg_mm_late"], bins)
# chrt_data

# Chart title
title_txt = f"Late Detraining Passengers: {STNS['chi']}"
title = ttl.format_title(chi_stats, title_txt)

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

# Quarterly stats
chi_qtr_stats = detrn.get_sum_stats_by_group(
    chi, [COLS["year"], COLS["quarter"]], AGG["columns"], AGG["funcs"], chi_trn_arrivals, chi_detrn
)
chi_qtr_stats.sort_values(by=[COLS["year"], COLS["quarter"]], ascending=[True, True])

# Write to file
filepath = parent_path.joinpath("data", "student", "stu-amtk-chi_qtr_stats.csv")
chi_qtr_stats.to_csv(filepath, index=True)

# Assemble the data for the chart
chrt_data = vis_bar.create_detrain_chart_frame(chi_qtr_stats, CHRT_BAR["columns"])

# Get station code, station name, city, and state to use in the chart title
text = frm.drop_dups_and_squeeze(
    chi, [COLS["station_code"], COLS["station"], COLS["city"], COLS["state"]]
)

# Chart title
title_txt = (
    f"Detraining Passengers: {text['Arrival Station']} ({text['Arrival Station Code']}), "
    f"{text['City']}, {text['State']}"
)
title = ttl.format_title(chi_stats, title_txt)

# Create and display grouped bar chart
chart = vis_bar.create_grouped_bar_chart(
    chrt_data,
    "Fiscal Period:N",
    "Passengers:Q",
    "Arrival Status:N",
    CHRT_BAR["xoffset_sort"],
    CHRT_BAR["colors"],
    title,
)

# chart.display()

# CHI: On-time performance metrics by service line
chi_svc_trns = chi.groupby(COLS["svc_line"]).size().reset_index()  # Includes rows with NaN
chi_svc_trns.columns = [COLS["svc_line"], COLS["trn_arrivals"]]
chi_svc_trns.sort_values(by=COLS["trn_arrivals"], ascending=False, inplace=True)
chi_svc_trns.reset_index(drop=True, inplace=True)

# Add train arrival ratios (year_qtr/total)
chi_svc_trns.loc[:, COLS["trn_arrival_ratio"]] = (
    chi_svc_trns[COLS["trn_arrivals"]] / chi_trn_arrivals
)

# Get summary stats by COLS["svc_line"]
chi_svc_line_stats = detrn.get_sum_stats_by_group(
    chi, COLS["svc_line"], AGG["columns"], AGG["funcs"]
)

# Merge train arrivals by service line
chi_svc_line_stats = chi_svc_line_stats.merge(chi_svc_trns, on=COLS["svc_line"], how="inner")

# Move train arrival columns
cols = chi_svc_line_stats.columns.tolist()
cols = [cols[0]] + cols[-2:] + cols[1:-2]
chi_svc_line_stats = chi_svc_line_stats[cols]

# Add service line detraining ratios
chi_svc_line_stats.loc[:, "Service Line Detraining Ratio"] = (
    chi_svc_line_stats["Total Detraining Customers sum"] / chi_detrn
)

# Move service line detraining ratio column
chi_svc_line_stats.insert(
    3, "Service Line Detraining Ratio", chi_svc_line_stats.pop("Service Line Detraining Ratio")
)

# Sort by passengers detrained (descending order)
chi_svc_line_stats.sort_values(by="Total Detraining Customers sum", ascending=False, inplace=True)

# Reset index
chi_svc_line_stats.reset_index(drop=True, inplace=True)

chi_svc_lines = chi.groupby(COLS["svc_line"])[[COLS["svc_line"], COLS["late_detrn_avg_mm_late"]]]
chrt_data = chi_svc_lines.apply(lambda x: x).reset_index(drop=True)  # Flatten for Altair

# Chart title
title_txt = (
    f"Late Detraining Passengers: {text['Arrival Station']} ({text['Arrival Station Code']}), "
    f"{text['City']}, {text['State']}"
)
title = ttl.format_title(chi_stats, title_txt)

# Create and display the box plots
chart = box.create_box_plot(
    chrt_data,
    "Late Detraining Customers Avg Min Late:Q",
    "Average Minutes Late",
    "Service Line:N",
    COLS["svc_line"],
    CHRT_BOX["y_axis"]["sort"],
    CHRT_BOX["colors"],
    title,
    CHRT_BOX["padding"],
)

# chart.display()

#3 Los Angeles Union Station (LAX), Los Angeles, CA

lax = ntwk.by_station(stations, "LAX")

# Train arrivals (total)
lax_trn_arrivals = lax.shape[0]

# Detraining totals
lax_detrn = lax[COLS["total_detrn"]].sum()
lax_detrn_late = lax[COLS["late_detrn"]].sum()
lax_detrn_on_time = lax_detrn - lax_detrn_late

# mean late arrival times summary statistics
lax_stats = detrn.get_sum_stats(lax, AGG["columns"], AGG["funcs"])

# Drop missing values
lax_avg_min_late = lax[COLS["late_detrn_avg_mm_late"]].dropna().reset_index(drop=True)

# Call the custom frm.describe_numeric_column() function again
lax_avg_min_late_describe = frm.describe_numeric_column(lax_avg_min_late)

# Visualize distribution of mean late arrival times
# Convert to DataFrame
lax_avg_min_late = lax_avg_min_late.to_frame(name=COLS["avg_mm_late"])

# Get mean and standard deviation
mu = lax_avg_min_late_describe["center"]["mean"]
sigma = lax_avg_min_late_describe["spread"]["std"]

# Get max value (for x-axis ticks); pad max value for chart display
max_val = lax_avg_min_late_describe["position"]["max"]
max_val_ceil = (np.ceil(max_val / 10) * 10).astype(int)

# Create bins
lax_min_late, bins, num_bins, bin_width = frm.create_bins(lax_avg_min_late, COLS["avg_mm_late"], 10)

# Bin the data
chrt_data = frm.bin_data(lax_min_late, COLS["avg_mm_late"], bins)
# chrt_data

# Chart title
title_txt = f"Late Detraining Passengers: {STNS['lax']}"
title = ttl.format_title(lax_stats, title_txt)

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

# On-time performance metrics (by fiscal year and quarter)
lax_qtr_stats = detrn.get_sum_stats_by_group(
    lax,
    [COLS["year"], COLS["quarter"]],
    AGG["columns"],
    AGG["funcs"],
    lax_trn_arrivals,
    lax_detrn,
)
lax_qtr_stats.sort_values(by=[COLS["year"], COLS["quarter"]], ascending=[True, True])

# Write to file
filepath = parent_path.joinpath("data", "student", "stu-amtk-lax_qtr_stats.csv")
lax_qtr_stats.to_csv(filepath, index=True)

# Visualize detraining passengers
# Assemble the data for the chart
chrt_data = vis_bar.create_detrain_chart_frame(lax_qtr_stats, CHRT_BAR["columns"])

# Get station code, station name, city, and state to use in the chart title
text = frm.drop_dups_and_squeeze(
    lax, [COLS["station_code"], COLS["station"], COLS["city"], COLS["state"]]
)

# Chart title
title_txt = (
    f"Detraining Passengers: {text['Arrival Station']} ({text['Arrival Station Code']}), "
    f"{text['City']}, {text['State']}"
)
title = ttl.format_title(lax_stats, title_txt)

# Create and display grouped bar chart
chart = vis_bar.create_grouped_bar_chart(
    chrt_data,
    "Fiscal Period:N",
    "Passengers:Q",
    "Arrival Status:N",
    CHRT_BAR["xoffset_sort"],
    CHRT_BAR["colors"],
    title,
)
# chart.display()

# On-time performance metrics by service line
lax_svc_trains = lax.groupby(COLS["svc_line"]).size().reset_index()  # Includes rows with NaN
lax_svc_trains.columns = [COLS["svc_line"], COLS["trn_arrivals"]]
lax_svc_trains.sort_values(by=COLS["trn_arrivals"], ascending=False, inplace=True)
lax_svc_trains.reset_index(drop=True, inplace=True)

# Add train arrival ratios (year_qtr/total)
lax_svc_trains.loc[:, COLS["trn_arrival_ratio"]] = (
    lax_svc_trains[COLS["trn_arrivals"]] / lax_trn_arrivals
)

# Get summary stats by COLS["svc_line"]
lax_svc_line_stats = detrn.get_sum_stats_by_group(
    lax, COLS["svc_line"], AGG["columns"], AGG["funcs"]
)

# Merge train arrivals by service line
lax_svc_line_stats = lax_svc_line_stats.merge(lax_svc_trains, on=COLS["svc_line"], how="inner")

# Move train arrival columns
cols = lax_svc_line_stats.columns.tolist()
cols = [cols[0]] + cols[-2:] + cols[1:-2]
lax_svc_line_stats = lax_svc_line_stats[cols]

# Add service line detraining ratios
lax_svc_line_stats.loc[:, "Service Line Detraining Ratio"] = (
    lax_svc_line_stats["Total Detraining Customers sum"] / lax_detrn
)

# Move service line detraining ratio column
lax_svc_line_stats.insert(
    3, "Service Line Detraining Ratio", lax_svc_line_stats.pop("Service Line Detraining Ratio")
)

# Sort by passengers detrained (descending order)
lax_svc_line_stats.sort_values(by="Total Detraining Customers sum", ascending=False, inplace=True)

# Reset index
lax_svc_line_stats.reset_index(drop=True, inplace=True)

# Visualize distribution of mean late arrival times
lax_svc_lines = lax.groupby(COLS["svc_line"])[[COLS["svc_line"], COLS["late_detrn_avg_mm_late"]]]
chrt_data = lax_svc_lines.apply(lambda x: x).reset_index(drop=True)  # Flatten for Altair

# Chart title
title_txt = (
    f"Late Detraining Passengers: {text['Arrival Station']} ({text['Arrival Station Code']}), "
    f"{text['City']}, {text['State']}"
)
title = ttl.format_title(lax_stats, title_txt)

chart = box.create_box_plot(
    chrt_data,
    "Late Detraining Customers Avg Min Late:Q",
    "Average Minutes Late",
    "Service Line:N",
    COLS["svc_line"],
    CHRT_BOX["y_axis"]["sort"],
    CHRT_BOX["colors"],
    title,
    CHRT_BOX["padding"],
)
# chart.display()

