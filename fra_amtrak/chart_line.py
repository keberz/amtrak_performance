import altair as alt


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


def configure_line(
    frame,
    x_shorthand,
    x_title,
    x_sort_order,
    y_shorthand,
    y_title,
    y_tick_count_max,
    point,
    color_shorthand,
    colors,
    tooltip_config,
):
    """Create a line chart.

    Parameters:
        frame (DataFrame): data frame
        x_shorthand (str): shorthand value for the x-axis
        x_title (str): x-axis title
        x_sort_order (str): sort order
        y_shorthand (str): shorthand value for the y-axis
        y_title (str): y-axis title
        y_tick_count_max (int): maximum y-axis tick count
        point (bool): point shape
        color_shorthand (str): shorthand value for the color
        colors (dict): color palette
        tooltip_config (list): nested dictionaries containing tooltip configuration values
\
    Returns:
        alt.Chart: line chart
    """

    return (
        alt.Chart(frame)
        .mark_line(point=point)
        .encode(
            x=configure_x_axis(x_shorthand, x_title, x_sort_order),
            y=configure_y_axis(y_shorthand, y_title, y_tick_count_max),
            color=configure_color(color_shorthand, colors),
            tooltip=configure_tooltip(tooltip_config),
        )
    )


def configure_line_dash(
    frame,
    x_shorthand,
    x_title,
    x_sort_order,
    y_shorthand,
    y_title,
    y_tick_count_max,
    point,
    stroke_dash,
    color_shorthand,
    colors,
    tooltip_config,
):
    """Create a line chart featuring a stroke dash style.

    Parameters:
        frame (DataFrame): data frame
        x_shorthand (str): shorthand value for the x-axis
        x_title (str): x-axis title
        x_sort_order (str): sort order
        y_shorthand (str): shorthand value for the y-axis
        y_title (str): y-axis title
        y_tick_count_max (int): maximum y-axis tick count
        point (bool): point shape
        stroke_dash (list): stroke dash style
        color_shorthand (str): shorthand value for the color
        colors (dict): color palette
        tooltip_config (list): nested dictionaries containing tooltip configuration values

    Returns:
        alt.Chart: line chart
    """

    return (
        alt.Chart(frame)
        .mark_line(point=point, strokeDash=stroke_dash)
        .encode(
            x=configure_x_axis(x_shorthand, x_title, x_sort_order),
            y=configure_y_axis(y_shorthand, y_title, y_tick_count_max),
            color=configure_color(color_shorthand, colors),
            tooltip=configure_tooltip(tooltip_config),
        )
    )


def configure_tooltip(config):
    """Returns a tooltip configuration object for a bar chart.

    Parameters:
        config (list): nested dictionaries containing tooltip configuration values

    Returns:
        alt.Chart: tooltip configuration object
    """

    return [
        alt.Tooltip(dict_["shorthand"], title=dict_["title"], format=dict_["format"])
        if dict_["format"]
        else alt.Tooltip(dict_["shorthand"], title=dict_["title"])
        for dict_ in config
    ]


def configure_x_axis(shorthand, title, sort_order):
    """Returns an alt.X object configured with the provided < shorthand > and
    other values.

    Parameters:
        shorthand (str): shorthand value for the x-axis
        title (str): x-axis title
        sort_order (str): sort order

    Returns:
        alt.X: x-axis object
    """

    return alt.X(
        shorthand=shorthand,
        axis=alt.Axis(
            labelAngle=270,
            labelFontWeight="normal",
            labelPadding=5,
            title=title,
            titleFontSize=10,
            titleFontWeight="bold",
        ),
        sort=sort_order,
    )


def configure_y_axis(shorthand, title, tick_count_max):
    """Returns an alt.Y object configured with the provided < shorthand > and
    other values.

    Parameters:
        shorthand (str): shorthand value for the y-axis
        title (str): y-axis title
        tick_count_max (int): maximum y-axis tick count

    Returns:
        alt.Y: y-axis object
    """

    return alt.Y(
        shorthand=shorthand,
        axis=alt.Axis(
            grid=True,
            labelAngle=0,
            labelFontWeight="normal",
            labelPadding=5,
            tickCount=20,
            title=title,
            titleFontSize=10,
            titleFontWeight="bold",
            values=list(range(0, tick_count_max, 5)),
        ),
        scale=alt.Scale(domain=[0, tick_count_max]),
    )


def create_line_chart(
    frame,
    x_shorthand,
    x_title,
    x_sort_order,
    y_shorthand,
    y_title,
    y_tick_count_max,
    point,
    color_shorthand,
    colors,
    tooltip_config,
    title,
    padding=10,
    height=300,
    width=600,
):
    """Creates a line chart.

    Parameters:
        frame (DataFrame): data frame
        x_shorthand (str): shorthand value for the x-axis
        x_title (str): x-axis title
        x_sort_order (str): sort order
        y_shorthand (str): shorthand value for the y-axis
        y_title (str): y-axis title
        y_tick_count_max (int): maximum y-axis tick count
        point (bool): point shape
        color_shorthand (str): shorthand value for the color
        colors (dict): color palette
        tooltip_config (list): nested dictionaries containing tooltip configuration values
        title (str): chart title
        padding (int): padding value
        height (int): chart height
        width (int): chart width

    Returns:
        alt.Chart: line chart
    """

    line = configure_line(
        frame,
        x_shorthand,
        x_title,
        x_sort_order,
        y_shorthand,
        y_title,
        y_tick_count_max,
        point,
        color_shorthand,
        colors,
        tooltip_config,
    )

    return alt.layer(line).properties(title=title, padding=padding, height=height, width=width)


def create_line_chart_interp(
    frame,
    x_shorthand,
    x_title,
    x_sort_order,
    y_shorthand,
    y_title,
    y_tick_count_max,
    point,
    color_shorthand,
    colors,
    tooltip_config,
    title,
    padding=10,
    height=300,
    width=600,
):
    """Creates a line chart featuring interpolated values.

    Parameters:
        frame (DataFrame): data frame
        x_shorthand (str): shorthand value for the x-axis
        x_title (str): x-axis title
        x_sort_order (str): sort order
        y_shorthand (str): shorthand value for the y-axis
        y_title (str): y-axis title
        y_tick_count_max (int): maximum y-axis tick count
        point (bool): point shape
        color_shorthand (str): shorthand value for the color
        colors (dict): color palette
        tooltip_config (list): nested dictionaries containing tooltip configuration values
        title (str): chart title
        padding (int): padding value
        height (int): chart height
        width (int): chart width

    Returns:
        alt.Chart: line chart
    """

    line = configure_line(
        frame,
        x_shorthand,
        x_title,
        x_sort_order,
        y_shorthand,
        y_title,
        y_tick_count_max,
        point,
        color_shorthand,
        colors,
        tooltip_config,
    )

    # Hack: copy frame, estimate missing values using linear interpolation
    frame_interp = frame.copy()
    frame_interp["Late Detraining Customers Avg Min Late mean"] = (
        frame.groupby("Train Number")["Late Detraining Customers Avg Min Late mean"]
        .apply(lambda group: group.interpolate())
        .reset_index(drop=True)
    )

    line_interp = configure_line_dash(
        frame_interp,
        x_shorthand,
        x_title,
        x_sort_order,
        y_shorthand,
        y_title,
        y_tick_count_max,
        False,
        [5, 5],
        color_shorthand,
        colors,
        tooltip_config,
    )

    return alt.layer(line, line_interp).properties(
        title=title, padding=padding, height=height, width=width
    )
