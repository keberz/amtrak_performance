import numpy as np
import pandas as pd


def add_stations_to_route(train, stations, station_order):
    """Adds < stations > to a train < route > before sorting the DataFrame by the specified
    < station_order >.

    Parameters:
        train (pd.DataFrame): DataFrame representing a train route
        stations (pd.DataFrame): DataFrame of train stations
        station_order (dict): dictionary of station codes and their order along the route

    Returns:
        pd.DataFrame: DataFrame representing a train route with added stations
    """

    # Add train number to < stations >
    stations.loc[:, "Train Number"] = train["Train Number"].iloc[0]

    return (
        pd.concat([train, stations], axis=0, ignore_index=True)
        .sort_values(by="Arrival Station Code", key=lambda x: x.map(station_order))
        .reset_index(drop=True)
    )


def by_service(stations, service, year=None, *quarters):
    """Return a DataFrame filtered by service and, optionally, year and zero to four specified
    quarters. Delegates to the function < filter_stations > the task of filtering the DataFrame.

    Parameters:
        stations (pd.DataFrame): DataFrame of train stations
        service (str): Passenger train service
        year (int): Fiscal year
        *quarters (int): One or more fiscal quarters

    Returns:
        pd.DataFrame: DataFrame filtered by service, year, and quarters
    """

    pass # TODO Implement me :)


def by_service_line(stations, service_line, year=None, *quarters):
    """Return a DataFrame filtered by service line and, optionally, year and zero to four specified
    quarters. Delegates to the function < filter_stations > the task of filtering the DataFrame.

    Parameters:
        stations (pd.DataFrame): DataFrame of train stations
        service_line (str): Passenger train service
        year (int): Fiscal year
        *quarters (int): One or more fiscal quarters

    Returns:
        pd.DataFrame: DataFrame filtered by service, year, and quarters
    """

    pass # TODO Implement me :)


def by_station(stations, station_code, year=None, *quarters):
    """Return a DataFrame filtered by a specified station and, optionally, year and zero to four
    specified quarters. Delegates to the function < filter_stations > the task of filtering the
    DataFrame.

    Parameters:
        stations (pd.DataFrame): DataFrame of train stations
        station_code (str): Station code
        year (int): Fiscal year
        *quarters (int): One or more fiscal quarters

    Returns:
        pd.DataFrame: DataFrame filtered by station code, year, and quarters
    """

    pass # TODO Implement me :)


def by_sub_service(stations, sub_service, year=None, *quarters):
    """Return a DataFrame filtered by sub_service and, optionally, year and zero to four specified
    quarters. Delegates to the function < filter_stations > the task of filtering the DataFrame.

    Parameters:
        stations (pd.DataFrame): DataFrame of train stations
        sub_service (str): Passenger train sub service
        year (int): Fiscal year
        *quarters (int): One or more fiscal quarters

    Returns:
        pd.DataFrame: DataFrame filtered by sub service, year, and quarters
    """

    pass # TODO Implement me :)


def by_train_number(stations, train_number, year=None, *quarters):
    """Return a DataFrame filtered by train_number and, optionally, year and zero to four specified
    quarters. Delegates to the function < filter_stations > the task of filtering the DataFrame.

    Parameters:
        stations (pd.DataFrame): DataFrame of train stations
        train_number (int): Train number
        year (int): Fiscal year
        *quarters (int): One or more fiscal quarters

    Returns:
        pd.DataFrame: DataFrame filtered by train number, year, and quarters
    """

    pass # TODO Implement me :)


def create_route(train, direction, station_order=None):
    """Create a DataFrame representing the route of a specifed train. Delegates to the function
    < sort_stations > the task of ordering the stations along the route given a specified
    < direction >.

    In certain cases, providing a < station_order > list is necessary to ensure that the DataFrame
    is ordered correctly. For example, the Michigan Wolverine route between Chicago and Detroit runs
    east-west but between Detroit and Pontiac the route runs northwest-southeast. Relying solely on
    longitude and latitude results in an incorrect station order.

    Parameters:
        train (pd.DataFrame): DataFrame of stations served by a particular train
        direction (str): Direction of the train
        station_order (dict): dictionary of station codes and their order along the route

    Returns:
        pd.DataFrame: DataFrame representing a train route
    """

    if station_order:
        train.sort_values(
            by="Arrival Station Code", key=lambda x: x.map(station_order), inplace=True
        )
        return train.reset_index(drop=True)

    match direction.lower():
        case "nb" | "northbound":
            columns, order = ["Latitude", "Longitude"], [True, True]
        case "sb" | "southbound":
            columns, order = ["Latitude", "Longitude"], [False, True]
        case "eb" | "eastbound":
            columns, order = ["Longitude", "Latitude"], [True, True]
        case "wb" | "westbound":
            columns, order = ["Longitude", "Latitude"], [False, True]
        case _:
            raise ValueError(
                "Direction invalid: choose eastbound, westbound, northbound, or southbound"
            )

    return train.sort_values(by=columns, ascending=order).reset_index(drop=True)


def filter_stations(stations, column=None, value=None, year=None, *quarters):
    """Return a DataFrame filtered by a < column > < value > pair, and optionally by year and
    between 0-4 specified quarters. A < year > must be provided to also filter by < quarters >.

    The pd.Series filtering mask is initialized to all True. If a < column > and < value > are
    provided, the mask is updated to reflect the new filtering criteria. If a < year > is provided,
    the mask is again updated. If < quarters > are provided, the mask is updated yet again. However,
    if the mask is never updated, the < stations > DataFrame is returned to the caller unchanged.

    WARN: when evaluating < value > check for None only; a truth value test is too strict,
    e.g., 0, 0.0, and False are valid column values.

    Parameters:
        stations (pd.DataFrame): DataFrame of train stations
        key (str): Column name to filter by
        value (str or int): Value to filter by in the specified column
        year (int): Fiscal year
        *quarters (int): One or more fiscal quarters

    Returns:
        pd.DataFrame: DataFrame filtered by specified criteria
    """

    mask = pd.Series([True] * len(stations))  # All True

    if column:
        if column not in stations.columns:
            raise ValueError("Invalid < column > name.")
        if value is None:
            raise ValueError("Invalid or missing < value >.")
        mask &= stations[column] == value

    if year:
        if not isinstance(year, int):
            raise TypeError("Invalid < year > type.")
        mask &= stations["Fiscal Year"] == year

        if quarters:
            if not all(isinstance(num, int) and num in (1, 2, 3, 4) for num in quarters):
                raise ValueError("Only 1-4 quarters can be specified.")
            mask &= stations["Fiscal Quarter"].isin(quarters)

    if mask.all():
        return stations  # mask remains all True
    else:
        return stations[mask].reset_index(drop=True)


def get_country(states_provinces, jurisdiction):
    """Evaluates the passed in <jurisdiction> against US states, Canadian provinces, and the US
    District of Columbia. If a match is obtained, the associated country name is returned to the
    caller. Otherwise, < np.nan > is returned.

    Parameters:
        states_provinces (dict): A dictionary containing lists of US and Canadian states, provinces,
                                 and the District of Columbia.
        jurisdiction (str): A state or province name.

    Returns:
        str: The country name associated with the passed in <jurisdiction>.
    """

    if jurisdiction in states_provinces["United States"]:
        return "United States"
    elif jurisdiction in states_provinces["Canada"]:
        return "Canada"
    else:
        return np.nan


def get_region_division(regions_divisions, jurisdiction):
    """Evaluates the passed in <jurisdiction> against US regions and divisions. If a match is
    obtained, the associated region and division is returned to the caller in a tuple. Otherwise,
    < np.nan >, < np.nan > is returned.

    Parameters:
        regions_divisions (dict): A dictionary containing US and Canadian regions and divisions.
        jurisdiction (str): A state, province, or district name.

    Returns:
        str: The region and division associated with the passed in <jurisdiction>.
    """

    for region, divisions in regions_divisions.items():
        for division, jurisdictions in divisions.items():
            if jurisdiction in jurisdictions:
                return region, division
    return np.nan, np.nan


def get_n_busiest_stations(stations, n=5, geo_unit=None, year=None, *quarters):
    """Return the n busiest stations by detraining passenger count, optionally filtered by
    < year > and zero to four specified < quarters >. Delegates to the function
    < get_nlargest() > the task of returning the n busiest stations.

    Parameters:
        stations (pd.DataFrame): DataFrame of train stations
        n (int): Number of stations to return
        geo_unit (str): Additional column to group by (e.g., "Region", "Division")
        year (int): Fiscal year
        *quarters (int): One or more fiscal quarters

    Returns:
        pd.DataFrame: DataFrame of the n busiest stations
    """

    stn_code = "Arrival Station Code"
    total_detrn = "Total Detraining Customers"

    if year:
        stations = filter_stations(stations, None, None, year, *quarters)

    # Group by column(s)
    groups = [geo_unit, stn_code] if geo_unit else [stn_code]

    # Group stations and sum detraining passenger counts
    total_psgr = stations.groupby(groups)[total_detrn].sum().reset_index()

    # WARN: DataFrame.apply(..., include_groups=False) to silence warning
    # DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns.
    if geo_unit:
        n_largest = total_psgr.groupby(geo_unit, group_keys=False).apply(
            get_nlargest, column=total_detrn, n_rows=n, include_groups=False
        )
    else:
        n_largest = get_nlargest(total_psgr, total_detrn, n)

    # Merge with the original DataFrame to include other columns like "Station Name"
    n_largest = n_largest.merge(
        stations.drop_duplicates(stn_code),
        on=stn_code,
        how="left",
        suffixes=("", "_dup"),
    )

    # Drop the duplicate COLS["total_detrn"] column
    n_largest = n_largest.drop(columns=[f"{total_detrn}_dup"])

    return n_largest


def get_nlargest(frame, column, n_rows=5):
    """Returns the n rows with the largest values in the specified column.

    Parameters:
       frame (pd.DataFrame): The DataFrame to be processed.
       column (str): The column to be used for ranking.
       n_rows (int): The number of rows to retain.

    Returns:
        pd.DataFrame: Largest n rows values in the specified column.
    """

    pass # TODO Implement me :)
