import numpy as np
import pandas as pd
import re
import scipy.stats as stats
import warnings


def aggregate_data(frame, columns, k=1.5):
    """Returns a DataFrame with aggregated statistics for the passed in columns.

    Parameters:
        frame (pd.DataFrame): DataFrame
        columns (list): list of columns to be aggregated
        k (float): constant for whisker calculation

    Returns:
        pd.DataFrame: DataFrame with aggregated statistics
    """

    # Compute aggregation statistics
    agg_stats = frame.groupby(columns[0])[[columns[1]]].describe().reset_index()

    # Rename columns
    agg_stats.columns = [columns[0], "count", "mean", "std", "min", "25%", "50%", "75%", "max"]

    # Compute interquartile range (IQR) and whisker boundaries
    agg_stats["iqr"] = agg_stats["75%"] - agg_stats["25%"]
    agg_stats["min_"] = agg_stats["25%"] - k * agg_stats["iqr"]
    agg_stats["max_"] = agg_stats["75%"] + k * agg_stats["iqr"]

    # Add whiskers
    data_points = frame.merge(agg_stats[[columns[0], "min_", "max_"]], on=columns[0], how="left")

    # Add lower bound
    mask_lower = data_points[columns[1]] >= data_points["min_"]
    agg_stats["lower"] = (
        data_points[mask_lower].groupby(columns[0])[columns[1]].min().reset_index(drop=True)
    )

    # Add upper bound
    mask_upper = data_points[columns[1]] <= data_points["max_"]
    agg_stats["upper"] = (
        data_points[mask_upper].groupby(columns[0])[columns[1]].max().reset_index(drop=True)
    )

    # Calculate outliers separately
    outliers = data_points[
        (data_points[columns[1]] < data_points["min_"])
        | (data_points[columns[1]] > data_points["max_"])
    ]
    outliers = outliers.groupby(columns[0])[columns[1]].apply(list).reset_index(name="outliers")

    # Initialize the 'outliers' column with empty lists and set dtype to object.
    agg_stats["outliers"] = [[] for _ in range(len(agg_stats))]
    agg_stats["outliers"] = agg_stats["outliers"].astype(object)

    # Merge to get the 'Color' column
    agg_stats = agg_stats.merge(
        frame[[columns[0], "Color"]].drop_duplicates(), on=columns[0], how="left"
    )

    # Combine and merge outliers within agg_stats DataFrame
    if "outliers" in agg_stats:
        agg_stats = agg_stats.merge(outliers, on=columns[0], how="left", suffixes=("", "_new"))
        agg_stats["outliers"] = agg_stats.apply(
            lambda row: row["outliers"]
            + (row["outliers_new"] if isinstance(row["outliers_new"], list) else []),
            axis=1,
        )
        agg_stats.drop(columns=["outliers_new"], inplace=True)

    return agg_stats


def bin_data(frame, column, bins):
    """Bins the passed in < column > data.

    Parameters:
        frame (pd.DataFrame): DataFrame of interest
        column (str): column to be binned
        bins (np.array): bins

    Returns:
        pd.DataFrame: binned data
    """

    # Count frequency per bin
    binned_data = frame.groupby("bin")[column].count().reset_index(name="count")

    # Compute bin start, end, and center
    binned_data["bin_start"] = binned_data["bin"].apply(lambda x: bins[x])
    # binned_data["bin_end"] = binned_data["bin"].apply(lambda x: bins[x + 1])
    binned_data["bin_end"] = binned_data["bin_start"] + 15
    binned_data["bin_center"] = (binned_data["bin_start"] + binned_data["bin_end"]) / 2

    return binned_data


def convert_column_to_frame(frame, column, drop_na=False, drop_index=True):
    """Converts a DataFrame < column > (a Series) to a DataFrame.

    Parameters:
        frame (pd.DataFrame): DataFrame of interest
        column (str): column to convert
        drop_na (bool): drop missing values
        drop_index (bool): insert index as column in the new DataFrame

    Returns:
        pd.DataFrame: DataFrame
    """

    return (
        pd.DataFrame(frame[column]).dropna().reset_index(drop=drop_index)
        if drop_na
        else pd.DataFrame(frame[column]).reset_index(drop=drop_index)
    )


def create_bins(frame, column, bin_width):
    """Creates bins for the passed in  < column > per the specified < bin_width >.

    Parameters:
        frame (pd.DataFrame): DataFrame of interest
        column (str): Column to be binned
        bin_width (int): Width of each bin

    Returns:
        tuple: binned pd.DataFrame, bins, num_bins, and bin_width
    """
    # Calculate the range for bins
    bin_min = np.floor(frame[column].min() / bin_width) * bin_width
    bin_max = np.ceil(frame[column].max() / bin_width) * bin_width + bin_width

    # Create bins
    bins = np.arange(bin_min, bin_max, bin_width)

    # Digitize data to discover each value's bin
    frame["bin"] = np.digitize(frame[column], bins) - 1

    # Ensure max value is in last bin
    frame.loc[frame[column] == bins[-1], "bin"] = len(bins) - 2

    # Return DataFrame with bins, bin edges, number of bins, and bin width
    return frame, bins, len(bins) - 1, bin_width


def describe_numeric_column(column):
    """Returns a dictionary of descriptive or summary statistics for for the passed in numeric
    < column >. If a non-numeric column is passed to the function a UserWarning is raised and
    None is returned to the caller.

    Parameters:
        column (pd.Series): column of interest

    Returns:
        dict: dictionary of descriptive statistics
    """

    try:
        if np.issubdtype(column.dtype, np.number):
            if column.isnull().any():
                warnings.warn(
                    "Column contains missing values. Interquartile range (IQR) will not be computed.",
                    UserWarning,
                )

            min = column.min()
            max = column.max()

            return {
                "type": column.__class__,
                "name": column.name,
                "values": {
                    "non_null": column.count(),
                    "missing": column.isna().sum(),
                    "dtype": column.dtype,
                },
                "center": {
                    "mean": column.mean(),
                    "median": column.median(),
                    "mode": column.mode()[0] if not column.mode().empty else None,
                },
                "position": {
                    "min": min,
                    "25%": column.quantile(0.25),
                    "50%": column.quantile(),
                    "75%": column.quantile(0.75),
                    "max": max,
                },
                "spread": {
                    "variance": column.var(),
                    "std": column.std(),
                    "range": max - min,
                    "iqr": stats.iqr(column),
                },
                "shape": {
                    "skewness": column.skew(),
                    "kurtosis": column.kurt(),
                },
            }
        else:
            warnings.warn("Provided column is not numeric.", UserWarning)

            return None

    except (AttributeError, TypeError, ValueError) as e:
        warnings.warn(f"{e}", UserWarning)

        return None


def drop_dups_and_squeeze(frame, columns):
    """Selects specified < columns > in < frame >, drops duplicate rows,
    and then squeezes the DataFrame into a Series to be returned to the
    caller.

    Parameters:
        frame (pd.DataFrame): DataFrame of interest
        columns (list): List of columns to retain

    Returns:
        pd.Series: Series object
    """

    pass  # TODO Implement me :)


def find_non_numeric_values(data, column):
    """Discover unique non-integer values in a column of a DataFrame. Creates a mask of non-integer
    values in the column and returns the unique non-integer values.

    Parameters
        data (DataFrame): The DataFrame containing the column to check.
        column (str): The name of the column to check for non-integer values.

    Returns
        list: A list of unique non-integer values found in the column.
    """

    mask = pd.to_numeric(data[column], errors="coerce")
    non_integer_values = data[column][mask.isna()]  # filter

    return list(non_integer_values.unique())


def normalize_string(value, pattern, replace=" "):
    """
    Normalize a single string < value >. Trim leading/trailing spaces and
    replace any substring matching the passed in regular expression < pattern >
    with the provided < replace > string.

    Parameters:
       value (str): String to normalize
       pattern(re.Pattern): Regular expression pattern to use
       replace (str): Replacement string for the pattern (default = single space)

    Returns:
         str: Normalized value
    """

    return re.sub(pattern, replace, value).strip() if isinstance(value, str) else value


def normalize_series_strings(series, pattern, replace=" "):
    """
    Normalize all strings in the passed in < series >. Delegate the task of normalizing each string
    to the function < normalize_string() >.

    Parameters:
        series (pd.Series): Series to normalize
        pattern (re.Pattern): Regular expression pattern to use
        replace (str): Replacement string for the pattern (default = single space)

    Returns:
        pd.Series: Cleaned column
    """

    return series.map(lambda value: normalize_string(value, pattern, replace))


def normalize_dataframe_strings(frame, pattern, replace=" "):
    """
    Normalize all string values in the passed in < frame >. Delegate the task of normalizing each
    column to the function < normalize_series_strings() >.

    Parameters:
       frame (pd.DataFrame): DataFrame to normalize
       pattern(re.Pattern): Regular expression pattern to use
       replace (str): Replacement string for the pattern (default = single space)

    Returns:
         pd.DataFrame: Cleaned DataFrame
    """

    return frame.apply(lambda column: normalize_series_strings(column, pattern, replace))
