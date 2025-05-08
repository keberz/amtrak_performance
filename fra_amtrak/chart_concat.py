import altair as alt

from enum import Enum


class Orient(Enum):
    VERTICAL = 0
    HORIZONTAL = 1


def concat_charts(charts, orient, columns=2, spacing=10, title=""):
    """Concatenates multiple charts into a single chart.

    Parameters:
        charts (list): list of altair.Chart objects
        orient (Orient): concatenated charts orientation (vertical or horizontal)
        columns (int): number of columns
        spacing (int): spacing between charts
        title (alt.TitleParams): chart title

    Returns:
        alt.Chart: concatenated chart
    """

    if orient == Orient.VERTICAL:
        concat = alt.vconcat(*charts)
    else:
        concat = alt.hconcat(*charts)

    print(f"Type of concat: {type(concat)}")

    return alt.ConcatChart(concat=[concat], columns=columns, spacing=spacing, title=title)

    # return alt.ConcatChart(
    #     concat=[alt.hconcat(*charts)],
    #     columns=3,
    #     spacing=10,
    #     title=alt.TitleParams(
    #         "Mega Ball, Pick 5 even/odd selections",
    #         align="center",
    #         anchor="middle",
    #         fontSize=12,
    #         fontWeight="bold",
    #     ),
    # )


def configure_legend(frame, orient, height=50, width=20):
    """Configures the chart legend.

    Parameters:
        chart (alt.Chart): chart object
        title (str): legend title
        orient (str): legend orientation

    Returns:
        alt.Chart: chart object with legend configured
    """

    return (
        alt.Chart(frame)
        .mark_rect()
        .encode(
            alt.Y("train:N", axis=alt.Axis(orient=orient, title=None)),
            color=alt.Color("color:N", scale=None, legend=None),
        )
        .properties(height=height, width=width)
    )


def create_layered_histogram(charts, legend, title):
    # Combine histograms
    layered_histogram = (
        alt.layer(*charts).resolve_scale(y="shared").properties(height=400, width=680)
    )

    # Add legend
    return alt.hconcat(layered_histogram, legend).properties(title=title, padding=15)
