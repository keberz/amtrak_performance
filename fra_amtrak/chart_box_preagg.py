import altair as alt

from enum import Enum


class Orient(Enum):
    VERTICAL = 0
    HORIZONTAL = 1


def configure_box_dimensions(base, size, color_shorthand, orient):
    """Returns an alt.Chart object with box boundaries configured.

    Parameters:
        base (alt.Chart): base chart object
        size (int): box size
        color_shorthand (str): box color
        orient (Orient): orientation of the box


    Returns:
        alt.Chart: chart object with box configured
    """

    if orient == Orient.HORIZONTAL:
        return base.mark_bar(size=size).encode(
            x=alt.X("25%"), x2="75%", color=alt.Color(color_shorthand).legend(None).scale(None)
        )
    return base.mark_bar(size=size).encode(
        y=alt.Y("25%"), y2="75%", color=alt.Color(color_shorthand).legend(None).scale(None)
    )


def configure_median_line(base, color, size, orient):
    """Returns an alt.Chart object with median line configured.

    Parameters:
        base (alt.Chart): base chart object
        color (str): color of the median line
        size (int): size of the median line
        position (str): position of the median line
        orient (Orient): orientation of the median line

    Returns:
        alt.Chart: chart object with median line configured
    """

    if orient == Orient.HORIZONTAL:
        return base.mark_tick(color=color, size=size).encode(x="50%")
    return base.mark_tick(color=color, size=size).encode(y="50%")


def configure_whiskers(base, title, orient):
    """Returns an alt.Chart object with whiskers configured.

    Parameters:
        base (alt.Chart): base chart object
        title (str): title for the axis
        orient (Orient): orientation of the whiskers

    Returns:
        alt.Chart: chart object with whiskers configured
    """

    if orient == Orient.HORIZONTAL:
        return base.mark_rule().encode(x=alt.X("lower:Q").title(title), x2="upper:Q")
    return base.mark_rule().encode(y=alt.Y("lower:Q").title(title), y2="upper:Q")


def create_boxplot(
    data,
    x_shorthand,
    x_title,
    y_shorthand,
    y_title,
    box_size,
    outlier_shorthand,
    color_shorthand,
    chart_title,
    padding=15,
    height=300,
    width=600,
    orient=Orient.HORIZONTAL,
):
    """Returns an alt.Chart object with one or more box plots. The orientation of each box plot is
    determined by the < orient > parameter.

    Parameters:
        data (pd.DataFrame): DataFrame
        x_shorthand (str): x-axis shorthand
        x_title (str): x-axis title
        y_shorthand (str): y-axis shorthand
        y_title (str): y-axis title
        box_size (int): size of the box
        outlier_shorthand (str): shorthand for outliers
        color_shorthand (str): shorthand for color
        chart_title (str): title of the chart
        padding (int): padding of the chart
        height (int): height of the chart
        width (int): width of the chart
        orient (Orient): orientation of the box plot(s)

    Returns:
        alt.Chart: chart object containing one or more box plots
    """

    base = alt.Chart(data).encode(
        x=alt.X(x_shorthand).axis(labelAngle=0).title(x_title),
        y=alt.Y(y_shorthand).sort("descending").title(y_title)
        if orient == Orient.HORIZONTAL
        else alt.Y(y_shorthand).sort("ascending").title(y_title),
        tooltip=[
            alt.Tooltip(y_shorthand).title(y_title)
            if orient == Orient.HORIZONTAL
            else alt.Tooltip(x_shorthand).title(x_title),
            alt.Tooltip("lower").title("Lower Whisker"),
            alt.Tooltip("25%").title("25% Quartile"),
            alt.Tooltip("50%").title("Median"),
            alt.Tooltip("75%").title("75% Quartile"),
            alt.Tooltip("upper").title("Upper Whisker"),
        ],
    )

    rules = configure_whiskers(base, x_title if orient == Orient.HORIZONTAL else y_title, orient)
    bars = configure_box_dimensions(base, box_size, color_shorthand, orient)
    ticks = configure_median_line(base, "#FFFFFF", box_size, orient)

    outliers = (
        base.transform_flatten(flatten=["outliers"])
        .mark_point(style="boxplot-outliers")
        .encode(
            x=alt.X(outlier_shorthand) if orient == Orient.HORIZONTAL else alt.Undefined,
            y=alt.Y(outlier_shorthand) if orient == Orient.VERTICAL else alt.Undefined,
            color=alt.Color(color_shorthand).legend(None).scale(None),
            tooltip=[
                alt.Tooltip(outlier_shorthand).title(f"{outlier_shorthand[:-3].title()} (mins)"),
                alt.Tooltip(y_shorthand if orient == Orient.HORIZONTAL else x_shorthand).title(
                    y_title if orient == Orient.HORIZONTAL else x_title
                ),
            ],
        )
    )

    # Assemble the chart layers
    chart_layer = alt.layer(rules, bars, ticks, outliers).properties(
        padding=padding, height=height, width=width, title=chart_title
    )

    return chart_layer
