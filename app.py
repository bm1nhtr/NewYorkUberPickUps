"""Streamlit story of Uber pickups in New York, April 2014."""

from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import pydeck as pdk
import seaborn as sns
import streamlit as st

DATA_PATH = Path(__file__).parent / "uber-raw-data-apr14.csv"
WEEKDAYS_EN = "Mon Tue Wed Thu Fri Sat Sun".split()
WEEKDAYS_FR = "Lun Mar Mer Jeu Ven Sam Dim".split()
WEEKDAYS_FULL = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

sns.set_theme(style="ticks")
plt.rcParams.update(
    {
        "axes.titlesize": 13,
        "axes.titleweight": "medium",
        "axes.labelsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#44403c",
        "xtick.color": "#292524",
        "ytick.color": "#292524",
        "text.color": "#1c1917",
        "axes.labelcolor": "#1c1917",
        "axes.titlecolor": "#1c1917",
    }
)


def get_dom(dt):
    return dt.day


def get_weekday(dt):
    return dt.weekday()


def get_hour(dt):
    return dt.hour


def fmt(n):
    return f"{int(n):,}"


def ratio(a, b):
    return f"{a / b:.1f}"


def save(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _scatter_frame(ax, lon, lat, color, size=0.8, alpha=0.4):
    ax.scatter(lon, lat, s=size, alpha=alpha, c=color, rasterized=True, linewidths=0)
    ax.set_xlim(-74.1, -73.8)
    ax.set_ylim(40.7, 40.9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


@st.cache_data(show_spinner=False)
def load_trips():
    df = pd.read_csv(DATA_PATH)
    df["Date/Time"] = pd.to_datetime(df["Date/Time"])
    df["day"] = df["Date/Time"].map(get_dom)
    df["weekday"] = df["Date/Time"].map(get_weekday)
    df["hour"] = df["Date/Time"].map(get_hour)
    return df


@st.cache_data(show_spinner=False)
def build_story():
    df = load_trips()

    by_date = df.groupby("day").size()
    by_weekday = df.groupby("weekday").size()
    by_hour = df.groupby("hour").size()
    day_weekday = df.groupby("day")["weekday"].first()
    cross = df.groupby(["weekday", "hour"]).size().unstack()
    peak_wd, peak_hr = cross.stack().idxmax()

    counts, xedges, yedges = np.histogram2d(
        df["Lon"],
        df["Lat"],
        bins=100,
        range=[[-74.1, -73.9], [40.5, 41]],
    )
    max_idx = np.unravel_index(np.argmax(counts), counts.shape)
    peak_lon = (xedges[max_idx[0]] + xedges[max_idx[0] + 1]) / 2
    peak_lat = (yedges[max_idx[1]] + yedges[max_idx[1] + 1]) / 2

    moments = {
        "wed17": df[(df["hour"] == 17) & (df["weekday"] == 2)],
        "sun17": df[(df["hour"] == 17) & (df["weekday"] == 6)],
        "wed21": df[(df["hour"] == 21) & (df["weekday"] == 2)],
        "wed5": df[(df["hour"] == 5) & (df["weekday"] == 2)],
    }

    images = {}

    ax = msno.bar(df, figsize=(9, 3.2), fontsize=11, color="#44403c")
    ax.set_title("Missing values — Uber April 2014")
    images["missing"] = save(ax.figure)

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.hist(df["day"], bins=30, rwidth=0.8, range=(0.5, 30.5))
    ax.set_title("Frequency by DoM - Uber - April 2014")
    ax.set_xlabel("Date of the month")
    ax.set_ylabel("Frequency")
    ax.set_xticks(range(1, 31))
    ax.tick_params(axis="x", labelsize=8)
    images["dom"] = save(fig)

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(by_date.index, by_date.values)
    ax.set_title("Line plot - Uber - April 2014")
    ax.set_xlabel("Days of the month")
    ax.set_ylabel("Frequency")
    ax.set_xticks(range(1, 31))
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlim(1, 30)
    images["line"] = save(fig)

    ordered = by_date.sort_values()
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.bar(range(1, 31), ordered.values, width=0.8)
    ax.set_xticks(range(1, 31))
    ax.set_xticklabels(ordered.index.astype(int), fontsize=8)
    ax.set_xlabel("Date of the month")
    ax.set_ylabel("Frequency")
    ax.set_title("Frequency by DoM - Uber - April 2014 (Sorted)")
    images["sorted"] = save(fig)

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.hist(df["hour"], bins=24, range=(-0.5, 24), rwidth=0.8)
    ax.set_xlabel("Hour of the day")
    ax.set_ylabel("Frequency")
    ax.set_title("Frequency by Hour - Uber - April 2014")
    ax.set_xticks(range(0, 24))
    ax.tick_params(axis="x", labelsize=8)
    images["hour"] = save(fig)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.hist(df["weekday"], bins=7, rwidth=0.8, range=(-0.5, 6.5))
    ax.set_xlabel("Day of the week")
    ax.set_ylabel("Frequency")
    ax.set_title("Frequency by Weekdays - Uber - April 2014")
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels(WEEKDAYS_EN)
    images["weekday"] = save(fig)

    fig, ax = plt.subplots(figsize=(12, 4.4))
    heat = sns.heatmap(cross, linewidths=0.5, ax=ax)
    heat.set_title("Heatmap by Hour and weekdays - Uber - April 2014", fontsize=15)
    heat.set_yticklabels(WEEKDAYS_FR, rotation="horizontal")
    heat.set_xlabel("Hour")
    heat.set_ylabel("")
    images["heatmap"] = save(fig)

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.hist(df["Lon"], bins=100, range=(-74.1, -73.9), color="g", alpha=0.5, label="Longitude")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Frequency")
    ax.set_title("Longitude - Uber - April 2014")
    images["lon"] = save(fig)

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.hist(df["Lat"], bins=100, range=(40.5, 41), color="r", alpha=0.5, label="Latitude")
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Frequency")
    ax.set_title("Lattitude - Uber - April 2014")
    images["lat"] = save(fig)

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_title("Longitude and Latitude distribution - Uber - April 2014", fontsize=15)
    ax.hist(
        df["Lon"],
        bins=100,
        range=(-74.1, -73.9),
        color="g",
        alpha=0.5,
        label="Longitude",
    )
    ax.legend(loc="best")
    ax2 = ax.twiny()
    ax2.hist(df["Lat"], bins=100, range=(40.5, 41), color="r", alpha=0.5, label="Latitude")
    ax2.legend(loc="upper left")
    images["twiny"] = save(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    _scatter_frame(ax, df["Lon"], df["Lat"], "#1d4ed8")
    ax.set_title("Scatter plot - Uber - April 2014")
    images["scatter"] = save(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    _scatter_frame(ax, df["Lon"], df["Lat"], "#1d4ed8")
    ax.scatter(
        peak_lon,
        peak_lat,
        color="red",
        s=200,
        marker="x",
        linewidths=3,
        label=f"Densest point ({peak_lon:.4f}, {peak_lat:.4f})",
        zorder=3,
    )
    wall_st_lon = (-74.02, -74.00)
    wall_st_lat = (40.700, 40.712)
    ax.add_patch(
        patches.Rectangle(
            (wall_st_lon[0], wall_st_lat[0]),
            wall_st_lon[1] - wall_st_lon[0],
            wall_st_lat[1] - wall_st_lat[0],
            linewidth=2,
            edgecolor="blue",
            facecolor="none",
            label="Wall Street / Financial District",
            zorder=3,
        )
    )
    ax.set_title("Scatter plot - Uber - April 2014")
    ax.legend(loc="best")
    images["scatter_annotated"] = save(fig)

    moment_titles = {
        "wed17": "Pickup density — Wed 5PM",
        "sun17": "Pickup density — SUN 17PM",
        "wed21": "Pickup density — Wed 21PM",
        "wed5": "Pickup density — Wed 5AM",
    }
    for key, title in moment_titles.items():
        rush = moments[key]
        fig, ax = plt.subplots(figsize=(6.2, 6.2))
        _scatter_frame(ax, rush["Lon"], rush["Lat"], "darkred", size=1, alpha=0.4)
        ax.set_title(title)
        images[key] = save(fig)

    busiest_day = int(by_date.idxmax())
    quiet_day = int(by_date.idxmin())
    second_day = int(by_date.drop(labels=[busiest_day]).idxmax())
    busiest_wd = int(by_weekday.idxmax())
    quiet_wd = int(by_weekday.idxmin())
    busiest_hour = int(by_hour.idxmax())
    quiet_hour = int(by_hour.idxmin())

    stats = {
        "n": int(len(df)),
        "missing": int(df.isna().sum().sum()),
        "busiest_day": busiest_day,
        "busiest_day_n": int(by_date.max()),
        "busiest_day_wd": int(day_weekday.loc[busiest_day]),
        "second_day": second_day,
        "second_day_n": int(by_date.loc[second_day]),
        "second_day_wd": int(day_weekday.loc[second_day]),
        "quiet_day": quiet_day,
        "quiet_day_n": int(by_date.min()),
        "quiet_day_wd": int(day_weekday.loc[quiet_day]),
        "sundays": [int(d) for d, wd in day_weekday.items() if int(wd) == 6],
        "sunday_counts": {
            int(d): int(by_date.loc[d]) for d, wd in day_weekday.items() if int(wd) == 6
        },
        "fridays": [int(d) for d, wd in day_weekday.items() if int(wd) == 4],
        "friday_counts": {
            int(d): int(by_date.loc[d]) for d, wd in day_weekday.items() if int(wd) == 4
        },
        "busiest_wd": busiest_wd,
        "busiest_wd_n": int(by_weekday.max()),
        "quiet_wd": quiet_wd,
        "quiet_wd_n": int(by_weekday.min()),
        "monday_n": int(by_weekday.loc[0]),
        "busiest_hour": busiest_hour,
        "busiest_hour_n": int(by_hour.max()),
        "quiet_hour": quiet_hour,
        "quiet_hour_n": int(by_hour.min()),
        "hour_21": int(by_hour.loc[21]),
        "hour_20": int(by_hour.loc[20]),
        "hour_7": int(by_hour.loc[7]),
        "hour_8": int(by_hour.loc[8]),
        "sat_peak_hr": int(cross.loc[5].idxmax()),
        "sat_peak_n": int(cross.loc[5].max()),
        "sun_0": int(cross.loc[6, 0]),
        "peak_wd": int(peak_wd),
        "peak_hr": int(peak_hr),
        "peak_cross": int(cross.loc[peak_wd, peak_hr]),
        "peak_lon": float(peak_lon),
        "peak_lat": float(peak_lat),
        "peak_bin": int(counts.max()),
        "moments": {key: int(len(part)) for key, part in moments.items()},
        "wall_n": int(
            (df["Lon"].between(-74.02, -74.00) & df["Lat"].between(40.700, 40.712)).sum()
        ),
    }
    return images, stats


def section(kicker, title):
    st.markdown(f'<p class="kicker">{kicker}</p>', unsafe_allow_html=True)
    st.markdown(f"## {title}")


def story(text):
    st.markdown(f'<p class="story">{text}</p>', unsafe_allow_html=True)


def caption(text):
    st.markdown(f'<p class="caption">{text}</p>', unsafe_allow_html=True)


PRESETS = {
    "All month": (list(range(7)), (0, 23)),
    "Weekday rush": ([0, 1, 2, 3, 4], (16, 19)),
    "Wednesday 17:00": ([2], (17, 17)),
    "Friday evening": ([4], (17, 23)),
    "Sunday": ([6], (0, 23)),
    "Before dawn": (list(range(7)), (0, 5)),
}


def playground(df):
    section("Play", "Look around before the reading")
    story(
        "Choose a weekday and an hour. The map redraws from the pickups that remain. "
        "The presets are the slices the charts below come back to: the evening rush, "
        "Wednesday at 17:00, Friday evening, Sunday, and the hours before dawn."
    )
    if "map_days" not in st.session_state:
        st.session_state.map_days = list(range(7))
    if "map_hours" not in st.session_state:
        st.session_state.map_hours = (0, 23)
    if "applied_preset" not in st.session_state:
        st.session_state.applied_preset = "All month"

    preset = st.pills(
        "Start from",
        list(PRESETS),
        selection_mode="single",
        default=st.session_state.applied_preset,
        key="map_preset_pills",
    )
    if preset and preset != st.session_state.applied_preset:
        st.session_state.applied_preset = preset
        st.session_state.map_days = list(PRESETS[preset][0])
        st.session_state.map_hours = PRESETS[preset][1]
        st.rerun()

    filters = st.columns(2)
    with filters[0]:
        days = st.multiselect(
            "Weekday",
            list(range(7)),
            format_func=lambda i: WEEKDAYS_FULL[i],
            key="map_days",
        )
    with filters[1]:
        hours = st.slider("Hour", 0, 23, key="map_hours")

    if not days:
        st.warning("Add at least one weekday.")
        return

    part = df[df["weekday"].isin(days) & df["hour"].between(int(hours[0]), int(hours[1]))]
    share = len(part) / len(df) * 100
    m1, m2 = st.columns(2)
    m1.metric("Pickups in this slice", fmt(len(part)))
    m2.metric("Share of April", f"{share:.1f}%")
    if part.empty:
        st.warning("Nothing falls in this slice. Widen the hour or add a weekday.")
        return

    step = 0.004
    grid = (
        part.assign(
            Lat=(part["Lat"] / step).round() * step,
            Lon=(part["Lon"] / step).round() * step,
        )
        .groupby(["Lat", "Lon"], as_index=False)
        .size()
        .rename(columns={"size": "pickups"})
    )
    grid["radius"] = np.clip(np.sqrt(grid["pickups"]) * 14, 45, 320)
    deck = pdk.Deck(
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=grid,
                get_position="[Lon, Lat]",
                get_radius="radius",
                radius_min_pixels=2,
                radius_max_pixels=28,
                get_fill_color=[194, 65, 12, 170],
                pickable=True,
            )
        ],
        initial_view_state=pdk.ViewState(
            latitude=40.74,
            longitude=-73.98,
            zoom=11,
            pitch=40,
        ),
        map_style="light",
        map_provider="carto",
        tooltip={"text": "{pickups} pickups"},
    )
    st.pydeck_chart(deck, height=520, width="stretch")
    caption(
        "Each circle is a small patch of the city. A larger circle means more pickups in the hours and days you selected."
    )


def main():
    st.set_page_config(
        page_title="Uber NYC, April 2014",
        page_icon="🚕",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container { max-width: 980px; padding-top: 1.6rem; padding-bottom: 4rem; }
        h1 { font-family: Georgia, "Times New Roman", serif; font-weight: 500; letter-spacing: -0.02em; }
        h2 { font-family: Georgia, "Times New Roman", serif; font-weight: 500; padding-top: .4rem; }
        .kicker {
            letter-spacing: .16em; text-transform: uppercase; font-size: .72rem;
            color: #9a3412; font-weight: 650; margin: 1.6rem 0 .15rem;
        }
        .lede { font-size: 1.18rem; line-height: 1.65; color: #1c1917; }
        .story { font-size: 1.05rem; line-height: 1.7; color: #292524; margin: .2rem 0 .8rem; }
        .caption {
            color: #44403c; font-size: .95rem; line-height: 1.55;
            border-left: 3px solid #c2410c; padding-left: .8rem; margin: .15rem 0 1.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### April, one city")
    st.sidebar.caption("Supervised by Mano Joseph MATHEW")
    st.sidebar.caption("Uber New York · April 2014")
    st.sidebar.markdown(
        "\n".join(
            [
                "Play with the map",
                "1. A complete table",
                "2. Day of the month",
                "3. The 24-hour rhythm",
                "4. The week",
                "5. Hour × weekday",
                "6. Longitude and latitude",
                "7. The pickup map",
                "8. Four moments",
            ]
        )
    )
    df = load_trips()

    st.markdown('<p class="kicker">Geospatial data analysis · April 2014</p>', unsafe_allow_html=True)
    st.title("One month of Uber in New York")
    st.markdown(
        """
        <p class="lede">This project is supervised by <b>Mano Joseph MATHEW</b>.</p>
        """,
        unsafe_allow_html=True,
    )
    story(
        "It reads one month of Uber pickups in New York and turns that table into a map of time and place: "
        "when people called a car, and where they were standing."
    )

    section("The dataset", "A Freedom of Information release")
    story(
        "The file comes from "
        "<a href='https://github.com/fivethirtyeight/uber-tlc-foil-response'>FiveThirtyEight’s Uber TLC FOIL response</a>. "
        "On 20 July 2015, FiveThirtyEight asked the New York City Taxi &amp; Limousine Commission for trip records "
        "under the Freedom of Information Law. The Commission sent the files in batches, as they were received on "
        "3 August, 15 September, and 22 September 2015."
    )
    story(
        "The repository holds more than 4.5 million Uber pickups from April to September 2014, "
        "another 14.3 million from January to June 2015, trip-level files for 10 other for-hire companies, "
        "and daily aggregates for 329 for-hire companies. "
        "FiveThirtyEight used the release in four stories on Uber, taxis, the outer boroughs, and rush-hour traffic."
    )
    story(
        f"This project uses one of the six monthly files in <code>uber-trip-data</code>: "
        f"<code>uber-raw-data-apr14.csv</code>, with <b>{fmt(len(df))}</b> pickups. "
        "Each row has four fields. <b>Date/Time</b> is the pickup time. "
        "<b>Lat</b> and <b>Lon</b> are the pickup coordinates. "
        "<b>Base</b> is the TLC base company affiliated with the pickup. "
        "In this April file the five bases are Unter (B02512), Hinter (B02598), Weiter (B02617), "
        "Schmecken (B02682), and Danach-NY (B02764)."
    )

    playground(df)

    with st.spinner("Rebuilding the charts from every pickup..."):
        images, s = build_story()

    sunday_bits = ", ".join(
        f"the {d}th ({fmt(n)})" for d, n in s["sunday_counts"].items()
    )
    friday_bits = ", ".join(
        f"the {d}th ({fmt(n)})" for d, n in s["friday_counts"].items()
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pickups", fmt(s["n"]))
    c2.metric("Busiest day", f"{s['busiest_day']} Apr")
    c3.metric("Busiest hour", f"{s['busiest_hour']}:00")
    c4.metric("Brightest cell", f"{WEEKDAYS_EN[s['peak_wd']]} {s['peak_hr']}h")

    section("01  ·  The data", "Four columns, every cell filled")
    story(
        f"The file <code>uber-raw-data-apr14.csv</code> has {fmt(s['n'])} rows and 4 columns: "
        f"<b>Date/Time</b>, <b>Lat</b>, <b>Lon</b>, and <b>Base</b>. "
        f"The missingno chart counts empty cells in each column: {s['missing']} empty cells. "
        "From Date/Time, the columns <code>day</code>, <code>weekday</code>, and <code>hour</code> are extracted, "
        "and every chart below counts the rows in each group."
    )
    st.image(images["missing"], width="stretch")
    caption("All four columns are complete. The height of each bar is the number of values present — the same as the number of rows in the file.")

    section("02  ·  Day of the month", "Weekends thin out, Fridays stay busy")
    story(
        "Pickups tend to drop at the weekend. "
        f"The short bars in the histogram are {sunday_bits} — "
        "every Sunday in April, seven days apart."
    )
    st.image(images["dom"], width="stretch")
    caption("Each bar is one calendar day. The repeated short bars are those four Sundays.")
    story(
        "Fridays carry a large number of pickups. "
        f"Across the month they are {friday_bits}. "
        "On the line, demand climbs into Friday and then falls through Saturday into Sunday. "
        "The 4th and the 25th sit among the busiest days of April."
    )
    st.image(images["line"], width="stretch")
    caption("Calendar order. The rises into the 4th, the 11th, the 18th, and the 25th are the Fridays.")
    story(
        f"Ranked from quietest to busiest, day {s['busiest_day']} leaves the rest of the month behind: "
        f"{fmt(s['busiest_day_n'])} pickups, "
        f"{ratio(s['busiest_day_n'], s['quiet_day_n'])} times the quietest Sunday. "
        f"That {WEEKDAYS_FULL[s['busiest_day_wd']]} is the last day of April. "
        "It may be payday, a day when people spend more freely and call a car to do it."
    )
    st.image(images["sorted"], width="stretch")
    caption(
        f"Days ordered from fewest pickups to most. Day {s['busiest_day']} is the bar that breaks away from the pack."
    )

    section("03  ·  Hour of the day", "The peak is the evening rush")
    story(
        "The busiest time of day is the evening rush, the hour people leave work. "
        f"The histogram is the evidence. The peak is <b>{s['busiest_hour']}:00</b>, "
        f"with {fmt(s['busiest_hour_n'])} pickups — "
        f"{ratio(s['busiest_hour_n'], s['quiet_hour_n'])} times the quietest hour, "
        f"{s['quiet_hour']}:00 ({fmt(s['quiet_hour_n'])}). "
        f"The climb starts after 15:00 and stays high through 21:00 "
        f"({fmt(s['hour_21'])} pickups, still above 20:00). "
        f"A smaller shoulder at 7:00 ({fmt(s['hour_7'])}) and 8:00 ({fmt(s['hour_8'])}) "
        "is the morning commute."
    )
    st.image(images["hour"], width="stretch")
    caption("Twenty-four bars, one per hour. The tallest bar is 17:00, the end of the workday.")

    section("04  ·  Day of the week", "Wednesday is the busiest weekday")
    story(
        f"<b>Wednesday</b> has the most pickups of any weekday, {fmt(s['busiest_wd_n'])}. "
        f"<b>Sunday</b> has the fewest, {fmt(s['quiet_wd_n'])}, about half of Wednesday — "
        "the same low end already visible on the 6th, the 13th, the 20th, and the 27th. "
        f"Monday is low as well ({fmt(s['monday_n'])}). "
        "Wednesday also takes some weight off the payday reading of the 30th. "
        "That date falls on a Wednesday, the busiest weekday, so part of its height is the day of the week itself."
    )
    st.image(images["weekday"], width="stretch")
    caption("Monday through Sunday. Wednesday is the tallest bar; Sunday is the shortest.")

    section("05  ·  The cross", "The bright cell was already named")
    story(
        "Put the hour next to the weekday and the bright cell is the one those two charts already point to: "
        f"<b>{WEEKDAYS_FULL[s['peak_wd']]} at {s['peak_hr']}:00</b>, "
        f"{fmt(s['peak_cross'])} pickups. The heatmap is that crossing, drawn out."
    )
    st.image(images["heatmap"], width="stretch")
    caption("Rows run from Monday (Lun) to Sunday (Dim). Columns are hours 0–23. Lighter cells mean more pickups.")

    section("06  ·  Two axes", "The city is packed into a narrow band")
    story(
        "Pickup longitudes compress into the range −74.1 to −73.9: "
        "the width of Manhattan and the streets just off either river. "
        "Latitude runs from 40.5 to 41, and the main mass sits between about 40.72 and 40.80 — "
        "from downtown up into Midtown. The two separate histograms each show a single peak, "
        "and <code>twiny()</code> places them on one figure so the shapes can be compared."
    )
    left, right = st.columns(2)
    with left:
        st.image(images["lon"], width="stretch")
    with right:
        st.image(images["lat"], width="stretch")
    caption("Green is longitude, red is latitude. Both lean into one cluster, with a thin tail of trips outside the core.")
    st.image(images["twiny"], width="stretch")
    caption(
        "The two histograms use different horizontal axes on the same figure. "
        "The bottom axis is longitude, the top axis is latitude. Height is still the number of pickups."
    )

    section("07  ·  The pickup map", "Does Wall Street actually fill up?")
    story(
        "The scatter is the shape of Manhattan, thick from north to south. "
        "The rectangular gap in the middle of the island is Central Park: pickups go around it."
    )
    st.image(images["scatter"], width="stretch")
    caption("Each dot is one pickup. The white rectangle in the middle of the island is Central Park.")
    story(
        "The open question is whether Wall Street, at the southern tip, is where people get into the car. "
        f"The blue box is that district. It holds <b>{fmt(s['wall_n'])}</b> pickups, "
        f"about {s['wall_n'] / s['n'] * 100:.0f}% of the month. "
        f"The red X is the densest cell, in Midtown East "
        f"({s['peak_lat']:.3f}, {s['peak_lon']:.3f}), and that much smaller patch already holds "
        f"{fmt(s['peak_bin'])}. Wall Street is on the map. The pile of pickups is the X."
    )
    st.image(images["scatter_annotated"], width="stretch")
    caption("Blue box: Wall Street and the Financial District. Red X: the densest cell, in Midtown East.")

    section("08  ·  The same frame", "Four hours on the same island")
    story(
        "Same map limits, four different hours. "
        f"Wednesday at 17:00 has {fmt(s['moments']['wed17'])} pickups. "
        f"At 21:00, {fmt(s['moments']['wed21'])} remain. "
        f"Sunday at 17:00 has {fmt(s['moments']['sun17'])}. "
        f"Wednesday at 05:00 has {fmt(s['moments']['wed5'])}."
    )
    top_l, top_r = st.columns(2)
    bot_l, bot_r = st.columns(2)
    panels = [
        (top_l, "wed17", f"Wednesday 17:00 · {fmt(s['moments']['wed17'])} pickups."),
        (top_r, "sun17", f"Sunday 17:00 · {fmt(s['moments']['sun17'])} pickups."),
        (bot_l, "wed21", f"Wednesday 21:00 · {fmt(s['moments']['wed21'])} pickups."),
        (bot_r, "wed5", f"Wednesday 05:00 · {fmt(s['moments']['wed5'])} pickups."),
    ]
    for col, key, text in panels:
        with col:
            st.image(images[key], width="stretch")
            caption(text)

    section("Close", "A workday, in Midtown")
    story(
        f"April thickens on Wednesday, at {s['peak_hr']}:00, in Midtown. "
        "Sunday and 05:00 are the same city with far fewer calls."
    )


if __name__ == "__main__":
    main()
