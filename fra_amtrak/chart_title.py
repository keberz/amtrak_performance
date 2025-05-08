def format_title(
    frame, title_text, multiline=True, anchor="middle", font_size=14, subtitle_font_size=12
):
    """Returns a formatted title for the chart.

    Returns:
        frame (pd.DataFrame): summary statistics DataFrame
        title_text (str): title of the chart
        multiline (bool): whether the title should be split into multiple lines
        anchor (str): anchor position
        font_size (int): font size
        subtitle_font_size (int): subtitle font size
    """

    detrain_total = int(frame["Total Detraining Customers sum"].sum())
    detrain_late = int(frame["Late Detraining Customers sum"].sum())
    detrain_late_pct = round(detrain_late / detrain_total * 100, 2)
    detrain_on_time = int(detrain_total - detrain_late)
    detrain_on_time_pct = round(detrain_on_time / detrain_total * 100, 2)
    mean_mins_late = frame["Late Detraining Customers Avg Min Late mean"].mean()

    return {
        "text": title_text.split("\n") if multiline else title_text,
        "subtitle": (
            f"total: {detrain_total:,}; "
            f"on time: {detrain_on_time:,} ({detrain_on_time_pct}%); "
            f"late: {detrain_late:,} ({detrain_late_pct}%) | "
            f"mean mins late: {mean_mins_late:.2f}"
        ),
        "anchor": anchor,
        "fontSize": font_size,
        "subtitleFontSize": subtitle_font_size,
    }
