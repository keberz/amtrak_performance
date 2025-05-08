import altair as alt


def configure_bar_text(frame, x_shorthand, y_shorthand, color):
    """Returns a text configuration object for a bar chart.

    Parameters:
        frame (pd.DataFrame): DataFrame of interest
        x_shorthand (str): shorthand value for the x-axis
        y_shorthand (str): shorthand value for the y-axis
        color (str): text color

    Returns:
        alt.Chart: text configuration object
    """

    return (
        alt.Chart(frame)
        .mark_text(align="center", baseline="bottom", fontSize=8, dy=-2)
        .encode(
            x=alt.X(shorthand=x_shorthand),
            y=alt.Y(shorthand=y_shorthand),
            text=alt.Text(shorthand=y_shorthand, format=","),
            color=alt.value(color),
        )
    )


def configure_color(shorthand, colors):
    """Returns a color configuration object for a bar chart.

    Parameters:
        shorthand (str): shorthand value
        colors (dict): color palette

    Returns:
        alt.Chart: color configuration object
    """

    return alt.Color(
        shorthand=shorthand,
        scale=alt.Scale(domain=list(colors.keys()), range=list(colors.values())),
        title=shorthand[:-2].title(),
    )


def configure_x_axis(shorthand):
    """Returns an alt.X object configured with the provided shorthand value.

    Parameters:
        shorthand (str): shorthand value for the x-axis

    Returns:
        alt.X: x-axis object
    """

    return alt.X(
        shorthand=shorthand,
        axis=alt.Axis(
            labelAngle=0,
            labelFontWeight="normal",
            labelPadding=5,
            title=shorthand[:-2].capitalize(),
            titleFontSize=10,
            titleFontWeight="bold",
        ),
    )


def configure_x_offset(shorthand, sort_order):
    """Returns an alt.XOffset object configured with the provided shorthand value and sort order.

    Parameters:
        shorthand (str): shorthand value for the x-offset
        sort_order (str): sort order

    Returns:
        alt.XOffset: x-offset object
    """

    return alt.XOffset(shorthand=shorthand, sort=sort_order)


def configure_y_axis(shorthand):
    """Returns an alt.Y object configured with the provided shorthand value.

    Parameters:
        shorthand (str): shorthand value for the y-axis

    Returns:
        alt.Y: y-axis object
    """

    return alt.Y(
        shorthand=shorthand,
        axis=alt.Axis(
            grid=False,
            labelAngle=0,
            labelFontWeight="normal",
            labelPadding=5,
            title=shorthand[:-2].capitalize(),
            titleFontSize=10,
            titleFontWeight="bold",
        ),
    )


def create_detrain_chart_frame(frame, columns):
    """Returns a reshaped DataFrame object based on the passed in < frame >. The < columns > list
    is used to reduce the DataFrame to only the columns of interest.

    Parameters:
        frame (pd.DataFrame): DataFrame of interest
        columns (dict): columns of interest

    Returns:
        pd.DataFrame: DataFrame object
    """

    chart_data = frame[columns.values()].copy()  # deep copy (avoid SettingWithCopyWarning)

    # X axis label: < year >Q< quarter > periods
    chart_data.loc[:, "Fiscal Period"] = (
        chart_data[columns["year"]].astype(str) + "Q" + chart_data[columns["quarter"]].astype(str)
    )

    # BREAK: forget to subtract Late from Total for On Time count
    chart_data.loc[:, "On Time"] = (
        chart_data[columns["total_detrain"]] - chart_data[columns["late_detrain"]]
    )

    # Rename column
    chart_data.rename(
        columns={
            columns["late_detrain"]: "Late",
        },
        inplace=True,
    )

    # Drop unnecessary columns
    chart_data.drop(
        columns=[columns["year"], columns["quarter"], columns["total_detrain"]], inplace=True
    )

    # Reshape DataFrame for Altair
    chart_data = chart_data.melt(
        id_vars="Fiscal Period",
        value_vars=["On Time", "Late"],
        var_name="Arrival Status",
        value_name="Passengers",
    )

    return chart_data


def create_grouped_bar_chart(
    frame,
    x_shorthand,
    y_shorthand,
    xoffset_shorthand,
    xoffset_sort_order,
    colors,
    title,
    padding=15,
    height=200,
    width=600,
):
    """Instantiates a grouped bar chart object configured with the provided data, axes shorthands,
    colors, and title.

    Parameters:
        frame (pd.DataFrame): DataFrame of interest
        x_shorthand (str): shorthand value for the x-axis
        y_shorthand (str): shorthand value for the y-axis
        xoffset_shorthand (str): shorthand value for the x-offset
        xoffset_sort_order (dict): sort order for the x-offset
        colors (dict): color palette
        title (dict): chart title (and optional subtitle)
        padding (int): chart padding
        height (int): chart height
        width (int): chart width

    Returns:
        alt.Chart: grouped bar chart object
    """

    base = alt.Chart(frame)
    bar = base.mark_bar().encode(
        x=configure_x_axis(x_shorthand),
        xOffset=configure_x_offset(xoffset_shorthand, xoffset_sort_order),
        y=configure_y_axis(y_shorthand),
        color=configure_color(xoffset_shorthand, colors),
        tooltip=alt.Tooltip([
            x_shorthand[:-2].title(),
            xoffset_shorthand[:-2].title(),
            y_shorthand[:-2].title(),
        ]),
    )
    # text = configure_bar_text(frame, x_shorthand, y_shorthand, "#000000")

    layered_chart = (
        alt.layer(bar)
        # alt.layer(bar, text)
        .configure_view(stroke=None, strokeWidth=0)
        .properties(title=title, padding=padding, height=height, width=width)
    )

    return layered_chart
