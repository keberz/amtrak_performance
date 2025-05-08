import altair as alt
import pandas as pd


def configure_bar(
    frame,
    x_shorthand,
    x_title,
    x_tick_count_max,
    y_shorthand,
    y_title,
    y_stack,
    num_bins,
    bin_width,
    bar_color,
    tooltip_config,
):
    """Returns a bar chart object.

    Parameters:
        frame (pd.DataFrame): DataFrame of interest
        x_shorthand (str): shorthand value for the x-axis
        x_title (str): title for the x-axis
        x_tick_count_max (int): maximum x-axis tick count
        y_shorthand (str): shorthand value for the y-axis
        y_title (str): title for the y-axis
        y_stack (str): stack value
        num_bins (int): number of bins
        bin_width (int): width of the bins
        bar_color (str): bar color
        tooltip_config (list): nested dictionaries containing tooltip config values

    Returns:
        alt.Chart: bar chart object
    """

    return (
        alt.Chart(frame)
        .mark_bar(binSpacing=0, color=bar_color, opacity=1)
        .encode(
            x=configure_x_axis(x_shorthand, x_title, num_bins, bin_width, x_tick_count_max),
            y=configure_y_axis(y_shorthand, y_title, y_stack),
            tooltip=configure_tooltip(tooltip_config),
        )
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

    sigma_pos = mu + n * sigma
    sigma_neg = mu - n * sigma

    if sigma_neg > 0:
        return configure_line(
            pd.DataFrame({line_title: [sigma_pos, sigma_neg]}), line_shorthand, color
        )
    else:
        return configure_line(
            pd.DataFrame({line_title: [sigma_pos]}), line_shorthand, color
        )

    # return configure_line(
    #     pd.DataFrame({line_title: [mu + n * sigma, mu - n * sigma]}), line_shorthand, color
    # )


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


def configure_x_axis(shorthand, title, max_bins, bin_width, x_tick_count_max):
    """Returns an alt.X object configured with the provided < shorthand > and
    other values.

    Parameters:
        shorthand (str): shorthand value for the x-axis
        title (str): x-axis title
        max_bins (int): maximum number of bins
        bin_width (int): width of the bins
        x_tick_count_max (int): maximum x-axis tick count

    Returns:
        alt.X: x-axis object
    """

    return alt.X(
        shorthand=shorthand,
        bin=alt.Bin(maxbins=max_bins, step=bin_width),
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


def create_histogram(
    frame,
    x_shorthand,
    x_title,
    y_shorthand,
    y_title,
    y_stack,
    line_shorthand,
    mu,
    sigma,
    num_bins,
    bin_width,
    x_tick_count_max,
    bar_color,
    mu_color,
    sigma_color,
    tooltip_config,
    title,
    padding=15,
    height=300,
    width=600,
):
    """Creates a histogram with mu and sigma lines. Data must be pre-binned before passing it to
    this function.

    Parameters:
        frame (pd.DataFrame): DataFrame of interest
        x_shorthand (str): shorthand value for the x-axis
        x_title (str): title for the x-axis
        y_shorthand (str): shorthand value for the y-axis
        y_title (str): title for the y-axis
        y_stack (str): stack value
        line_shorthand (str): shorthand value for the line
        mu (float): mean value
        sigma (float): standard deviation value
        num_bins (int): number of bins
        bin_width (int): width of the bins
        x_tick_count_max (int): maximum x-axis tick count
        bar_color (str): color for the bars
        mu_color (str): color for the mu line
        sigma_color (str): color for the sigma lines
        tooltip_config (list): nested dictionaries containing tooltip config values
        title (str): chart title
        padding (int): chart padding
        height (int): chart height
        width (int): chart width

    Returns:
        alt.Chart: histogram with mu and sigma lines
    """

    bar = configure_bar(
        frame,
        x_shorthand,
        x_title,
        x_tick_count_max,
        y_shorthand,
        y_title,
        y_stack,
        num_bins,
        bin_width,
        bar_color,
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
