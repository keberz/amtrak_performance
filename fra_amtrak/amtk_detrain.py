def assign_color(fiscal_quarter, colors):
    """Returns a color from < colors > based on the passed in
    < fiscal quarter >. The fiscal quarter format is < year >Q< quarter > (e.g., 2024Q3).

    Parameters:
        fiscal_quarter (str): fiscal quarter
        colors (list): color palette

    Returns:
        str: color
    """

    return colors[0] if int(fiscal_quarter[-1]) % 2 == 0 else colors[1]


def compute_sum_stats(frame, agg_columns, agg_funcs, precision=4):
    """Computes summary statistics by constructing a dictionary of aggregation functions to be
    applied to specified < frame > columns. The < agg_columns > specifies the columns
    to aggregate, while the < agg_funcs > list specifies the types of functions to apply to each
    target column.

    The function assembles a dictionary of aggregation functions to apply to the specified columns
    by iterating over < frame.columns > and mapping the < agg_funcs > list to each column name key,
    if, and only if, the column is a member of < agg_columns >.

    The dictionary is passed directly to the DataFrame.agg() method to compute the summary
    statistics. The < precision > value determines the number of decimal places to retain.

    Parameters:
        frame (pd.DataFrame): DataFrame of interest
        agg_columns (list): List of columns to aggregate
        agg_funcs (list): List of aggregation functions to compute
        precision (int): Number of decimal places in which to round the summary stats

    Returns:
        pd.DataFrame: DataFrame of summary statistics for detraining passengers
    """

    pass  # TODO Implement me :)


def compute_sum_stats_by_group(frame, groups, agg_columns, agg_funcs, reset_idx=True, precision=4):
    """Performs a group by operation on the < frame > and then computes summary statistics for each
    group. The < groups > parameter specifies how to split < frame >. The < agg_columns > specifies
    the columns to aggregate, while the < agg_funcs > list specifies the types of functions to apply
    to each target column.

    The function assembles a dictionary of aggregation functions to apply to the specified columns
    by iterating over < frame.columns > and mapping the < agg_funcs > list to a each column name
    key, if, and only if, the column is a member of < agg_columns >.

    The dictionary is passed directly to the DataFrameGroupBy.agg() method to compute the summary
    statistics. The < precision > value determines the number of decimal places to retain. The
    index is also reset if < reset_idx > is True.

    Parameters:
        frame (pd.DataFrame): DataFrame of interest
        groups (func|dict|list|str): specifies how to group the stations
        columns (list): List of columns to aggregate
        agg_funcs (list): List of aggregation functions to compute
        reset_idx (bool): Whether to reset the index of the resulting DataFrame
        precision (int): Number of decimal places in which to round the summary stats

    Returns:
        pd.DataFrame: DataFrame of summary statistics for detraining passengers
    """

    pass # TODO Implement me :)


def flatten_columns(frame):
    """Flattens multi-index columns in the passed in < frame >. If a column is a tuple, the elements
    are joined by a space.

    Parameters:
        frame (pd.DataFrame): DataFrame of interest

    Returns:
        list: List of column names
    """

    return [
        col[0] if not col[1] else " ".join(col) if isinstance(col, tuple) else col
        for col in frame.columns
    ]


def format_year_quarter(row):
    """Combines the fiscal year and fiscal quarter row values for display purposes (e.g., x-axis or
    y-axis ticks/labels). Guard against float values being inserted into the formatted string
    by converting each value to an integer before joining them. The letter 'Q' is used to separate
    the fiscal year and quarter values.

    Format:
        < fiscal year >Q< fiscal quarter > e.g., 2024Q3

    Parameters:
        row (pd.Series): Series object containing fiscal year and quarter values

    Returns:
        str: Formatted string of fiscal year and quarter
    """

    pass # TODO Implement me :)


def get_late_to_total_detrain_ratio(frame, precision=4):
    """Computes the ratio of late to total detraining passengers for each station in the passed in
    < frame >. The < precision > value determines the number of decimal places to retain.

    Parameters:
        frame (pd.DataFrame): DataFrame of interest
        precision (int): Number of decimal places in which to round the computed ratio

    Returns:
        pd.Series: Series of late to total detraining passengers ratios
    """

    return (frame["Late Detraining Customers sum"] / frame["Total Detraining Customers sum"]).round(
        precision
    )


def get_mean_min_late(frame, precision=4):
    """Computes the mean late arrival time (minutes) for late detraining passengers. The
    < precision > value determines the number of decimal places to retain.

    Parameters
        frame (pd.DataFrame): DataFrame of interest
        precision (int): Number of decimal places to retain

    Returns
        pd.Series: Series of mean late arrival times for late detraining passengers
    """

    pass # TODO Implement me :)


def get_mean_min_late_by_groups(frame, groups=None, precision=4):
    """Computes the mean late arrival time (minutes) for late detraining passengers. Rounds the
    mean to the specified number of decimal places. Resets the index and renames the column before
    returning the DataFrame to the caller.

    Parameters
        frame (pd.DataFrame): DataFrame of interest
        groups (func|dict|list|str): specifies how to group the stations
        precision (int): Number of decimal places in which to round the mean late arrival times

    Returns
        pd.DataFrame|pd.Series: DataFrame or Series of mean late arrival times for late detraining
        passengers for each
    """

    return (
        frame.groupby(groups)["Late Detraining Customers Avg Min Late"]
        .mean()
        .round(precision)
        .reset_index()
        .rename(
            columns={
                "Late Detraining Customers Avg Min Late": "Late Detraining Customers Avg Min Late mean"
            }
        )
    )


def get_qtr_avg_min_late(frame, columns, column, colors):
    """Gathers the quarterly average late arrival times for detraining passengers. The resulting
    DataFrame is flattened and the fiscal year and quarter are joined to create a new column. The
    DataFrame is then reordered and alternating colors are assigned to each row.

    Parameters:
        frame (pd.DataFrame): DataFrame of interest
        columns (list): List of columns to aggregate
        column (str): New column name
        colors (list): Color palette

    Returns:
        pd.DataFrame: DataFrame of quarterly average late arrival times for detraining passengers
    """

    # Group by fiscal year and quarter and flatten
    avg_min_late = frame.groupby(columns[:2])[columns].apply(lambda x: x).reset_index(drop=True)

    # Add column
    avg_min_late.loc[:, column] = avg_min_late.apply(format_year_quarter, axis=1)

    # Drop columns and reorder
    avg_min_late.drop(columns[:2], axis=1, inplace=True)
    avg_min_late.insert(0, column, avg_min_late.pop(column))

    # Assign alternating colors
    avg_min_late.loc[:, "Color"] = avg_min_late[column].apply(assign_color, colors=colors)

    return avg_min_late


def get_route_sum_stats(frame, groups, agg_columns, agg_funcs, columns):
    """Computes summary statistics for detraining passengers for specified stations along a train
    route < groups >. The types of summary statistics to compute are specified by the < agg_funcs >
    parameter. Two additional metrics are also provided: the late to total detraining ratio and the
    mean late arrival time for late detraining passengers. Returns a reshaped DataFrame based on the
    passed-in < columns > augmented with columns of summary statistics. The ration of late to total
    detraining passengers is also calculated and rounded to five decimal places.

    Parameters:
        frame (pd.DataFrame): DataFrame of interest
        groups (func|dict|list|str): specifies how to group the stations
        agg_columns (list): List of columns to aggregate
        agg_funcs (list): List of aggregation functions to compute
        columns (list): List of <route> columns to retain along with the three new columns

    Returns:
        pd.DataFrame: DataFrame of station averages along the route
    """

    stats = get_sum_stats_by_group(frame, groups, agg_columns, agg_funcs)

    # Merge
    stats = (
        frame.loc[:, columns]
        .drop_duplicates()
        .merge(stats, on=groups, how="left")
        .reset_index(drop=True)
    )

    return stats


def get_sum_stats(frame, agg_columns, agg_funcs, precision=4):
    """Computes summary statistics for detraining passengers. The types of summary statistics to
    compute are specified by the < agg_funcs > parameter. Two additional metrics are also provided:
    the late to total detraining ratio and the mean late arrival time for late detraining
    passengers. The < precision > value determines the number of decimal places to retain.

    Parameters:
        frame (pd.DataFrame): DataFrame of interest
        groups (func|dict|list|str): specifies how to group the stations
        agg_columns (list): List of columns to aggregate
        agg_funcs (list): List of aggregation functions to compute

    Returns:
        pd.DataFrame: DataFrame of summary statistics for detraining passengers
    """

    # Summarize
    stats = compute_sum_stats(frame, agg_columns, agg_funcs)  # Do not reset index

    # Flatten
    stats = stats.unstack().to_frame().transpose()
    # print(f"system stats: {stats}")

    # Rename
    stats.columns = [f"{col} {idx}" for col, idx in stats.columns]
    # print(f"stats.columns: {stats.columns}")

    # Add Train Arrivals (every row represents a train arrival at a station)
    stats.loc[:, "Train Arrivals"] = frame.shape[0]
    stats.insert(0, "Train Arrivals", stats.pop("Train Arrivals"))  # Move

    # Compute late to total detraining passengers ratio
    stats.loc[:, "Late to Total Detraining Customers Ratio"] = get_late_to_total_detrain_ratio(
        stats
    )

    # Compute mean late arrival time for late detraining passengers
    stats.loc[:, "Late Detraining Customers Avg Min Late mean"] = get_mean_min_late(frame)

    # Add Total On Time Detraining Customers
    stats.loc[:, "Total On Time Detraining Customers sum"] = (
        stats["Total Detraining Customers sum"] - stats["Late Detraining Customers sum"]
    )

    # Reset index (to single-row DataFrame)
    stats.reset_index(drop=True, inplace=True)

    return stats


def get_sum_stats_by_group(
    frame, groups, agg_columns, agg_funcs, total_arrivals=None, total_detrain=None
):
    """Computes summary statistics for detraining passengers for specified < groups >. The types of
    summary statistics to compute are specified by the < agg_funcs > parameter. Six additional
    metrics are also provided: the late to total detraining ratio, the mean late arrival time for
    late detraining passengers, the total on time detraining customers, and, optionally, the train
    arrival ratio, and the detraining ratio.

    Parameters:
        frame (pd.DataFrame): DataFrame of interest
        groups (func|dict|list|str): specifies how to group the stations
        agg_columns (list): List of columns to aggregate
        agg_funcs (list): List of aggregation functions to compute
        total_arrivals (int): Total number of train arrivals
        total_detrain (int): Total number of detraining passengers

    Returns:
        pd.DataFrame: DataFrame of summary statistics for detraining passengers
    """
    # Aggregate
    stats = compute_sum_stats_by_group(frame, groups, agg_columns, agg_funcs)

    # Flatten multi-index columns
    stats.columns = flatten_columns(stats)

    # Add Train Arrivals (every row represents a train arrival at a station) and move
    train_arrivals = get_train_arrivals_by_group(frame, groups)
    stats = stats.merge(train_arrivals, on=groups, how="inner")
    stats.insert(
        stats.columns.get_loc("Total Detraining Customers sum"),
        "Train Arrivals",
        stats.pop("Train Arrivals"),
    )

    # Compute late to total detraining passengers ratio
    stats.loc[:, "Late to Total Detraining Customers Ratio"] = get_late_to_total_detrain_ratio(
        stats
    )

    # Add mean late arrival time for late detraining passengers
    mean_min_late = get_mean_min_late_by_groups(frame, groups)
    stats = stats.merge(mean_min_late, on=groups, how="inner")

    # Add Total On Time Detraining Customers
    stats.loc[:, "Total On Time Detraining Customers sum"] = (
        stats["Total Detraining Customers sum"] - stats["Late Detraining Customers sum"]
    )

    # Compute train arrival ratios (year_qtr/total)
    if total_arrivals:
        stats.loc[:, "Train Arrival Ratio"] = stats["Train Arrivals"] / total_arrivals
        # stats.loc[:, "Train Arrival Ratio"] = get_train_arrival_ratio(stats, total_arrivals)

        # Move
        stats.insert(
            stats.columns.get_loc("Total Detraining Customers sum"),
            "Train Arrival Ratio",
            stats.pop("Train Arrival Ratio"),
        )

    # Add service line detraining ratios
    if total_detrain:
        stats.loc[:, "Detraining Ratio"] = stats["Total Detraining Customers sum"] / total_detrain

        # Move
        stats.insert(
            stats.columns.get_loc("Total Detraining Customers sum"),
            "Detraining Ratio",
            stats.pop("Detraining Ratio"),
        )

    return stats


def get_train_arrival_ratio(frame, total_arrivals):
    """Computes the ratio of < frame > column train arrivals to < total_arrivals >.

    Parameters:
        frame (pd.DataFrame): DataFrame of interest
        total_arrivals (int): Total number of train arrivals

    Returns:
        pd.Series: Series of train arrival ratios (no percentage conversion)
    """

    pass # TODO Implement me :)


def get_train_arrivals_by_group(frame, groups):
    """Computes the total number of station train arrivals for each of the passed in < groups >.

    Parameters:
        frame (pd.DataFrame): DataFrame of interest
        groups (func|dict|list|str): specifies how to group the stations

    Returns:
        pd.DataFrame: DataFrame of total train arrivals for each station
    """

    train_arrivals = frame.groupby(groups).size().reset_index()
    train_arrivals.rename(columns={0: "Train Arrivals"}, inplace=True)

    return train_arrivals


def predict_avg_min_late_by_distance(result, distance_mi):
    """Given a specified < distance > predicts the average minutes late for late detraining
    passengers based on the computed linear regression < result >.

    Parameters:
        result (pd.Series): Series of linear regression coefficients
        distance_mi (int): distance in miles

    Returns:
        float: Predicted average minutes late for late detraining passengers
    """

    pass # TODO Implement me :)
