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
        legend=None,
    )


def configure_x_axis(shorthand, title):
    """Returns an alt.X object configured with the provided shorthand value.

    Parameters:
        shorthand (str): shorthand value for the x-axis
        title (str): title for the x-axis

    Returns:
        alt.X: x-axis object
    """

    return alt.X(
        shorthand=shorthand,
        axis=alt.Axis(
            labelAngle=0,
            labelFontWeight="normal",
            labelPadding=5,
            # title=title,
            titleFontSize=10,
            titleFontWeight="bold",
        ),
        scale=alt.Scale(zero=False),
        title = title
    )


def configure_y_axis(shorthand, title, sort):
    """Returns an alt.Y object configured with the provided shorthand value.

    Parameters:
        shorthand (str): shorthand value for the y-axis
        title (str): title for the y-axis
        sort (list): sort the entities of interest

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
            # title=title,
            titleFontSize=10,
            titleFontWeight="bold",
        ),
        sort=sort,
        title=title
    )


def create_box_plot(
    frame,
    x_shorthand,
    x_title,
    y_shorthand,
    y_title,
    y_sort,
    colors,
    title,
    padding=15,
    height=125,
    width=600,
):
    """Instantiates a box chart object configured with the provided data, axes shorthands, colors, and
    title.

    Parameters:
        frame (DataFrame): data frame
        x_shorthand (str): shorthand value for the x-axis
        x_title (str): title for the x-axis
        y_shorthand (str): shorthand value for the y-axis
        y_title (str): title for the y-axis
        y_sort (list): sort the entities of interest
        colors (dict): color palette
        title (str): title for the chart
        padding (int): padding
        height (int): height
        width (int): width

    Returns:
        alt.Chart: box chart object
    """

    base = alt.Chart(frame)
    box = base.mark_boxplot(size=20).encode(
        x=configure_x_axis(x_shorthand, x_title),
        y=configure_y_axis(y_shorthand, y_title, y_sort),
        color=configure_color(y_shorthand, colors),
        tooltip=[
            alt.Tooltip(y_shorthand, title=y_shorthand[:-2].title()),
            alt.Tooltip(x_shorthand, title=x_shorthand[-14:-2].title()),
        ],
    )

    chart = box.configure_view(stroke=None, strokeWidth=0).properties(
        title=title,
        padding=padding,
        width=width,
        height=height,
    )

    return chart
