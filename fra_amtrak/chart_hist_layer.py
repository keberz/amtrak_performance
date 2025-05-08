import altair as alt
import pandas as pd


def configure_bar(
    chart,
    x_shorthand,
    x_title,
    x_tick_count_max,
    x2_shorthand,
    y_shorthand,
    y_title,
    y_stack,
    max_bins,
    order_shorthand,
    color_shorthand,
    colors,
    tooltip_config,
):
    """Returns a bar chart object.

    Parameters:
        chart (alt.Chart): chart object
        x_shorthand (str): shorthand value for the x-axis
        x_title (str): x-axis title
        x_tick_count_max (int): maximum x-axis tick count
        x2_shorthand (str): shorthand value for the x2 channel
        y_shorthand (str): shorthand value for the y-axis
        y_title (str): y-axis title
        y_stack (str): stack value
        max_bins (int): maximum number of bins
        order_shorthand (str): shorthand value for the order
        color_shorthand (str): shorthand value for the color
        colors (dict): color palette
        tooltip_config (list): nested dictionaries containing tooltip config values

    Returns:
        alt.Chart: bar chart object
    """

    return chart.mark_bar(binSpacing=0, opacity=1).encode(
        x=configure_x_axis(x_shorthand, x_title, max_bins, x_tick_count_max),
        x2=alt.X2(x2_shorthand),
        y=configure_y_axis(y_shorthand, y_title, y_stack),
        color=configure_color(color_shorthand, colors),
        order=alt.Order(order_shorthand),
        tooltip=configure_tooltip(tooltip_config),
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


def configure_line(frame, x_shorthand, color):
    """Returns a line configuration object for a line chart.

    Parameters:
        frame (pd.DataFrame): DataFrame of interest
        x_shorthand (str): shorthand value for the x-axis
        color (str): line color

    Returns:
        alt.Chart: line configuration object
    """

    return alt.Chart(frame).mark_rule(color=color).encode(x=x_shorthand)


def configure_mu_line(line_shorthand, line_title, mu, color):
    """Returns a mu line object.

    Parameters:
        line_shorthand (str): shorthand value for the line
        line_title (str): title for the line
        mu (float): mean value
        color (str): line color

    Returns:
        alt.Chart: mu line object
    """

    return configure_line(pd.DataFrame({line_title: [mu]}), line_shorthand, color)


def configure_sigma_lines(line_shorthand, line_title, mu, sigma, color, n=1):
    """Returns sigma line objects.

    Parameters:
        line_shorthand (str): shorthand value for the line
        line_title (str): title for the line
        sigma (float): standard deviation value
        color (str): line color
        n (int): the sigma level (e.g., 1-sigma, 2-sigma)

    Returns:
        alt.Chart: sigma line object
    """

    return configure_line(
        pd.DataFrame({line_title: [mu + n * sigma, mu - n * sigma]}), line_shorthand, color
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


def configure_x_axis(shorthand, title, max_bins, x_tick_count_max):
    """Returns an alt.X object configured with the provided < shorthand > and
    other values.

    Parameters:
        shorthand (str): shorthand value for the x-axis
        title (str): x-axis title
        max_bins (int): maximum number of bins
        x_tick_count_max (int): maximum x-axis tick count

    Returns:
        alt.X: x-axis object
    """

    return alt.X(
        shorthand=shorthand,
        bin="binned",
        axis=alt.Axis(
            labelAngle=0,
            labelFontWeight="normal",
            labelPadding=5,
            tickCount=max_bins,
            title=title,
            titleFontSize=10,
            titleFontWeight="bold",
            values=list(range(0, x_tick_count_max + 15, 5)),
        ),
        scale=alt.Scale(domain=[0, x_tick_count_max]),
    )


def configure_y_axis(shorthand, title, stack=False):
    """Returns an alt.Y object configured with the provided < shorthand > and
    other values.

    Parameters:
        shorthand (str): shorthand value for the y-axis
        title (str): y-axis title
        stack (str): stack value

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
            title=title,
            titleFontSize=10,
            titleFontWeight="bold",
        ),
        stack=stack,
    )


def create_layered_histogram(
    frame,
    x_shorthand,
    x_title,
    x_tick_count_max,
    x2_shorthand,
    y_shorthand,
    y_title,
    y_stack,
    line_shorthand,
    mu,
    sigma,
    max_bins,
    bin_step,
    hst_order_shorthand,
    hst_color_shorthand,
    hst_colors,
    mu_color,
    sigma_color,
    tooltip_config,
    title,
    padding=10,
    height=300,
    width=600,
):
    """Creates a layered histogram.

    Paremeters:
        frame (pd.DataFrame): DataFrame of interest
        x_shorthand (str): shorthand value for the x-axis
        x_title (str): x-axis title
        x_tick_count_max (int): maximum x-axis tick count
        x2_shorthand (str): shorthand value for the x2 channel
        y_shorthand (str): shorthand value for the y-axis
        y_title (str): y-axis title
        y_stack (str): stack value
        line_shorthand (str): shorthand value for the line
        mu (float): mean value
        sigma (float): standard deviation value
        max_bins (int): maximum number of bins
        bin_step (int): bin step size
        hst_order_shorthand (str): shorthand value for the order
        hst_color_shorthand (str): shorthand value for the color
        hst_colors (dict): color palette
        mu_color (str): mu line color
        sigma_color (str): sigma line color
        tooltip_config (list): nested dictionaries containing tooltip config values
        title (str): chart title
        padding (int): padding value
        height (int): chart height
        width (int): chart width

    Returns:
        alt.Chart: layered histogram
    """

    chrt_data = transform_data(
        frame,
        x_shorthand,
        x_title,
        x2_shorthand,
        hst_order_shorthand,
        hst_color_shorthand,
        max_bins,
        bin_step,
    )
    bar = configure_bar(
        chrt_data,
        x_shorthand,
        x_title,
        x_tick_count_max,
        x2_shorthand,
        y_shorthand,
        y_title,
        y_stack,
        max_bins,
        hst_order_shorthand,
        hst_color_shorthand,
        hst_colors,
        tooltip_config,
    )

    # mu (μ) and sigma (σ) lines (μ ± σ, μ ± 2σ)
    line_title = line_shorthand[:-2].title()
    mu_line = configure_mu_line(line_shorthand, line_title, mu, mu_color)
    sigma_lines = configure_sigma_lines(line_shorthand, line_title, mu, sigma, sigma_color)
    two_sigma_lines = configure_sigma_lines(line_shorthand, line_title, mu, sigma, sigma_color, 2)

    return alt.layer(bar, mu_line, sigma_lines, two_sigma_lines).properties(
        title=title, padding=padding, height=height, width=width
    )

    # return chart.properties(title=title, padding=padding, height=height, width=width)


def transform_data(
    frame, x_shorthand, x_title, x2_shorthand, order_shorthand, color_shorthand, max_bins, bin_step
):
    """Transforms the chart data for display purposes.

    Parameters:
        frame (pd.DataFrame): DataFrame of interest
        x_shorthand (str): shorthand value for the x-axis
        x_title (str): x-axis title
        x2_shorthand (str): shorthand value for the x2 channel
        order_shorthand (str): shorthand value for the order
        color_shorthand (str): shorthand value for the color
        max_bins (int): maximum number of bins
        bin_step (int): bin step size

    Returns:
        alt.Chart: new chart object
    """

    # Remove encoding shorthand suffixes (e.g., ':N')
    x = x_shorthand[:-2]
    x2 = x2_shorthand[:-2]
    color = color_shorthand[:-2]
    order = order_shorthand[:-2]

    # JavsScript expression: 'datum.bin_start + " - " + datum.bin_end'
    bin_range = f"datum.{x} + ' - ' + datum.{x2}"

    return (
        alt.Chart(frame)
        .transform_aggregate(
            count="count()", groupby=[x, x2, color, order], mean_late=f"mean({x_title})"
        )
        .transform_bin([x, x2], x_title, bin=alt.Bin(maxbins=max_bins, step=bin_step))
        .transform_calculate(bin_range=bin_range)
    )
