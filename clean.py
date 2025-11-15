import os
import glob
import json
import random
from typing import List, Tuple
from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 9

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

DEFAULT_DATA_FOLDER = os.environ.get(
    "SPOTIFY_HISTORY_FOLDER",
    str(Path.home() / "Downloads" / "Spotify Extended Streaming History"),
)

LOCAL_LLM_ENABLED = os.environ.get("LOCAL_LLM_ENABLED", "true").lower() == "true"
LOCAL_LLM_ENDPOINT = os.environ.get(
    "LOCAL_LLM_ENDPOINT",
    "http://localhost:11434/api/generate",
)
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "llama3")

MIN_MS_PLAYED = 30 * 1000
TOP_N = 15
YEARLY_TOP_N = 10
BAR_COLOR = "#6a9e65"

TASTE_LABELS = [
    "Obsessive stan energy",
    "Balanced explorer",
    "Playlist goldfish",
    "Nostalgic loop gremlin",
]


def load_streaming_history(folder_path: str) -> pd.DataFrame:
    patterns = [
        os.path.join(folder_path, "Streaming_History_Audio_*.json"),
        os.path.join(folder_path, "Streaming_History_Video_*.json"),
    ]

    file_paths: List[str] = []
    for pattern in patterns:
        file_paths.extend(glob.glob(pattern))

    if not file_paths:
        raise FileNotFoundError(
            "No files found matching Streaming_History_Audio_*.json or "
            "Streaming_History_Video_*.json in the provided folder."
        )

    all_records = []

    print("Files being loaded:")
    for path in sorted(file_paths):
        print("  -", os.path.basename(path))
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    print(
                        f"Warning: {path} does not contain a list "
                        "at the top level, skipping."
                    )
                    continue
                all_records.extend(data)
            except json.JSONDecodeError as e:
                print(f"Error reading {path}: {e}. Skipping this file.")

    if not all_records:
        raise ValueError("No valid records found in any JSON files.")

    df = pd.DataFrame(all_records)

    for col in [
        "ts",
        "ms_played",
        "master_metadata_track_name",
        "master_metadata_album_artist_name",
        "master_metadata_album_album_name",
    ]:
        if col not in df.columns:
            df[col] = None

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["year"] = df["ts"].dt.year
    df["ms_played"] = pd.to_numeric(df["ms_played"], errors="coerce").fillna(0)

    return df


def filter_valid_music(df: pd.DataFrame) -> pd.DataFrame:
    mask_music = (
        df["master_metadata_track_name"].notna()
        & df["master_metadata_album_artist_name"].notna()
    )
    mask_duration = df["ms_played"] >= MIN_MS_PLAYED
    return df[mask_music & mask_duration].copy()


def ms_to_hours(ms: pd.Series) -> pd.Series:
    return ms / 1000 / 60 / 60


def plot_barh(series: pd.Series, title: str, xlabel: str, output_filename: str):
    plt.figure(figsize=(10, 6))
    series = series.sort_values(ascending=True)
    series.plot(kind="barh", color=BAR_COLOR)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=200)
    plt.close()
    print(f"Saved plot: {output_filename}")


def generate_insights(df: pd.DataFrame, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)

    artist_ms = (
        df.groupby("master_metadata_album_artist_name")["ms_played"]
        .sum()
        .sort_values(ascending=False)
        .head(TOP_N)
    )
    artist_hours = ms_to_hours(artist_ms)
    plot_barh(
        artist_hours,
        title=f"Top {TOP_N} Artists by Listening Time",
        xlabel="Listening time (hours)",
        output_filename=os.path.join(output_folder, "top_artists.png"),
    )

    track_ms = (
        df.groupby("master_metadata_track_name")["ms_played"]
        .sum()
        .sort_values(ascending=False)
        .head(TOP_N)
    )
    track_hours = ms_to_hours(track_ms)
    plot_barh(
        track_hours,
        title=f"Top {TOP_N} Tracks by Listening Time",
        xlabel="Listening time (hours)",
        output_filename=os.path.join(output_folder, "top_tracks.png"),
    )

    album_ms = (
        df.groupby("master_metadata_album_album_name")["ms_played"]
        .sum()
        .sort_values(ascending=False)
        .head(TOP_N)
    )
    album_hours = ms_to_hours(album_ms)
    plot_barh(
        album_hours,
        title=f"Top {TOP_N} Albums by Listening Time",
        xlabel="Listening time (hours)",
        output_filename=os.path.join(output_folder, "top_albums.png"),
    )


def print_top_series(label: str, series: pd.Series):
    print(f"  {label}:")
    if series.empty:
        print("    No data.")
        return

    max_value = series.max()
    for rank, (name, value) in enumerate(series.items(), start=1):
        bar_length = int((value / max_value) * 20) if max_value > 0 else 0
        bar = "#" * bar_length
        hours_str = f"{value:.2f} h"
        print(f"    {rank:2d}. {name}  [{hours_str}] {bar}")


def generate_yearly_terminal_insights(df: pd.DataFrame):
    df_years = df[df["year"].notna()].copy()
    if df_years.empty:
        print("No valid year data available for yearly insights.")
        return

    years = sorted(df_years["year"].dropna().unique())
    print()
    print("Yearly listening insights")
    print("=========================")

    for year in years:
        year_int = int(year)
        df_y = df_years[df_years["year"] == year_int]

        if df_y.empty:
            continue

        print()
        print(f"Year {year_int}")
        print("----------------------------------------")

        artist_ms = (
            df_y.groupby("master_metadata_album_artist_name")["ms_played"]
            .sum()
            .sort_values(ascending=False)
            .head(YEARLY_TOP_N)
        )
        artist_hours = ms_to_hours(artist_ms)
        print_top_series("Top artists", artist_hours)

        track_ms = (
            df_y.groupby("master_metadata_track_name")["ms_played"]
            .sum()
            .sort_values(ascending=False)
            .head(YEARLY_TOP_N)
        )
        track_hours = ms_to_hours(track_ms)
        print_top_series("Top tracks", track_hours)

        album_ms = (
            df_y.groupby("master_metadata_album_album_name")["ms_played"]
            .sum()
            .sort_values(ascending=False)
            .head(YEARLY_TOP_N)
        )
        album_hours = ms_to_hours(album_ms)
        print_top_series("Top albums", album_hours)


def generate_yearly_charts(df: pd.DataFrame, output_folder: str):
    df_years = df[df["year"].notna()].copy()
    if df_years.empty:
        print("No valid year data available for yearly charts.")
        return

    years = sorted(df_years["year"].dropna().unique())
    os.makedirs(output_folder, exist_ok=True)

    def make_yearly_chart(group_col: str, title: str, filename: str):
        max_hours_global = 0.0
        per_year_series = {}

        for year in years:
            year_int = int(year)
            df_y = df_years[df_years["year"] == year_int]
            if df_y.empty:
                continue
            ms = (
                df_y.groupby(group_col)["ms_played"]
                .sum()
                .sort_values(ascending=False)
                .head(YEARLY_TOP_N)
            )
            hours = ms_to_hours(ms)
            per_year_series[year_int] = hours
            if not hours.empty:
                max_hours_global = max(max_hours_global, float(hours.max()))

        if not per_year_series:
            print(f"No data for yearly chart: {title}")
            return

        n_years = len(per_year_series)
        fig_height = max(3, 2 * n_years)
        fig, axes = plt.subplots(
            nrows=n_years,
            ncols=1,
            figsize=(10, fig_height),
            sharex=True,
        )

        if n_years == 1:
            axes = [axes]

        for ax, (year_int, series) in zip(axes, sorted(per_year_series.items())):
            series_sorted = series.sort_values(ascending=True)
            ax.barh(series_sorted.index, series_sorted.values, color=BAR_COLOR)
            ax.set_ylabel(str(year_int))
            ax.grid(axis="x", linestyle=":", alpha=0.3)
            ax.tick_params(axis="y", labelsize=7)

        axes[0].set_title(title)
        axes[-1].set_xlabel("Listening time (hours)")

        plt.xlim(0, max_hours_global * 1.1 if max_hours_global > 0 else 1)

        plt.tight_layout()
        full_path = os.path.join(output_folder, filename)
        plt.savefig(full_path, dpi=200)
        plt.close()
        print(f"Saved yearly chart: {full_path}")

    make_yearly_chart(
        group_col="master_metadata_album_artist_name",
        title=f"Yearly Top {YEARLY_TOP_N} Artists by Listening Time",
        filename="yearly_top_artists.png",
    )

    make_yearly_chart(
        group_col="master_metadata_track_name",
        title=f"Yearly Top {YEARLY_TOP_N} Tracks by Listening Time",
        filename="yearly_top_tracks.png",
    )

    make_yearly_chart(
        group_col="master_metadata_album_album_name",
        title=f"Yearly Top {YEARLY_TOP_N} Albums by Listening Time",
        filename="yearly_top_albums.png",
    )


def build_yearly_comment(
    df_y: pd.DataFrame,
    top_artists_hours: pd.Series,
    top_tracks_hours: pd.Series,
) -> str:
    total_ms = df_y["ms_played"].sum()
    total_hours_series = ms_to_hours(pd.Series([total_ms]))
    total_hours = float(total_hours_series.iloc[0]) if total_ms > 0 else 0.0

    unique_artists = int(df_y["master_metadata_album_artist_name"].nunique())
    unique_tracks = int(df_y["master_metadata_track_name"].nunique())
    unique_albums = int(df_y["master_metadata_album_album_name"].nunique())

    lines = []

    if total_hours == 0:
        return "No listening data for this year."
    elif total_hours < 50:
        lines.append(f"Light listening year with about {total_hours:.1f} hours of music.")
    elif total_hours < 200:
        lines.append(
            f"Moderate listening year with roughly {total_hours:.1f} hours of music."
        )
    else:
        lines.append(f"Heavy listening year with around {total_hours:.1f} hours of music.")

    lines.append(
        f"Variety check: {unique_artists} artists, "
        f"{unique_albums} albums, {unique_tracks} tracks."
    )

    if total_ms > 0 and not top_artists_hours.empty:
        top_artist = top_artists_hours.index[0]
        top_artist_hours = float(top_artists_hours.iloc[0])
        top_artist_ms = top_artist_hours * 1000 * 60 * 60
        top_share = top_artist_ms / total_ms

        if top_share > 0.5:
            lines.append(
                f"{top_artist} completely ruled this year, taking more than "
                "half of your listening time."
            )
        elif top_share > 0.35:
            lines.append(
                f"{top_artist} was the clear comfort artist, but you still "
                "had room for others."
            )
        elif top_share > 0.2:
            lines.append(
                f"{top_artist} led the pack without totally dominating your vibe."
            )
        else:
            lines.append(
                "No single artist dominated, which points to a very exploratory mood."
            )

    total_plays = len(df_y)
    if total_plays > 0 and unique_tracks > 0:
        plays_per_track = total_plays / unique_tracks
        if plays_per_track > 4:
            lines.append("You leaned hard on repeat listening and replayed favorites a lot.")
        elif plays_per_track < 2:
            lines.append(
                "You kept cycling through new songs instead of looping the same ones."
            )
        else:
            lines.append("You balanced replaying favorites with trying new tracks.")

    if not top_tracks_hours.empty:
        top_track = top_tracks_hours.index[0]
        lines.append(f"Most played track: {top_track}.")

    return "\n".join(lines[:5])


def compute_year_features(
    df_y: pd.DataFrame,
    top_artists_hours: pd.Series,
    top_tracks_hours: pd.Series,
) -> Tuple[list, dict]:
    total_ms = df_y["ms_played"].sum()
    total_hours_series = ms_to_hours(pd.Series([total_ms]))
    total_hours = float(total_hours_series.iloc[0]) if total_ms > 0 else 0.0

    unique_artists = float(df_y["master_metadata_album_artist_name"].nunique())
    unique_tracks = float(df_y["master_metadata_track_name"].nunique())
    unique_albums = float(df_y["master_metadata_album_album_name"].nunique())

    total_plays = float(len(df_y))
    plays_per_track = total_plays / unique_tracks if unique_tracks > 0 else 0.0

    top_artist_share = 0.0
    if total_ms > 0 and not top_artists_hours.empty:
        top_artist_hours = float(top_artists_hours.iloc[0])
        top_artist_ms = top_artist_hours * 1000 * 60 * 60
        top_artist_share = top_artist_ms / total_ms

    top_track_hours = float(top_tracks_hours.iloc[0]) if not top_tracks_hours.empty else 0.0
    diversity_ratio = unique_tracks / total_plays if total_plays > 0 else 0.0

    features = [
        total_hours / 500.0,
        unique_artists / 500.0,
        unique_tracks / 1000.0,
        unique_albums / 500.0,
        top_artist_share,
        plays_per_track / 10.0,
        top_track_hours / 200.0,
        diversity_ratio,
    ]

    info = dict(
        total_hours=total_hours,
        unique_artists=unique_artists,
        unique_tracks=unique_tracks,
        unique_albums=unique_albums,
        top_artist_share=top_artist_share,
        plays_per_track=plays_per_track,
        top_track_hours=top_track_hours,
        diversity_ratio=diversity_ratio,
    )
    return features, info


def classify_taste_archetype(features: list) -> str:
    if not TORCH_AVAILABLE:
        top_artist_share = features[4]
        plays_per_track = features[5] * 10.0
        diversity_ratio = features[7]

        if top_artist_share > 0.45 and plays_per_track > 3.5:
            return "Obsessive stan energy"
        if diversity_ratio > 0.5 and plays_per_track < 2.5:
            return "Balanced explorer"
        if diversity_ratio < 0.3 and plays_per_track < 2.0:
            return "Playlist goldfish"
        return "Nostalgic loop gremlin"

    tensor = torch.tensor(features, dtype=torch.float32)
    torch.manual_seed(42)
    weight = torch.randn(len(features), len(TASTE_LABELS))
    bias = torch.randn(len(TASTE_LABELS))
    scores = tensor @ weight + bias
    probs = torch.softmax(scores, dim=0)
    idx = int(torch.argmax(probs).item())
    return TASTE_LABELS[idx]


def simple_roast_fallback(
    year_int: int,
    artist_hours: pd.Series,
    track_hours: pd.Series,
    archetype: str,
) -> str:
    top_artist = artist_hours.index[0] if not artist_hours.empty else "no one in particular"
    top_track = track_hours.index[0] if not track_hours.empty else "some mystery track"

    templates = [
        "Year {year}: You unlocked full {arch} mode. "
        "You basically hired {top_artist} as your emotional consultant while "
        "{top_track} looped like a loading screen.",
        "Year {year}: That was peak {arch}. "
        "Every time life blinked at you, you answered with more {top_artist} "
        "and another round of {top_track}.",
        "Year {year}: Your vibe screamed {arch}. "
        "{top_artist} handled the main storyline while {top_track} played "
        "every time you opened your brain.",
        "Year {year}: Statistically speaking, this was a cry for help from your queue. "
        "{arch}, powered mostly by {top_artist} and the eternal echoes of {top_track}.",
        "Year {year}: If Spotify did personality tests, yours would just be a screenshot of "
        "{top_artist}, {top_track}, and a giant label saying {arch}.",
    ]

    template = random.choice(templates)
    return template.format(year=year_int, arch=archetype, top_artist=top_artist, top_track=top_track)


def local_llm_roast(
    year_int: int,
    artist_hours: pd.Series,
    track_hours: pd.Series,
    comment_text: str,
    archetype: str,
    feature_info: dict,
) -> str:
    if not LOCAL_LLM_ENABLED or not REQUESTS_AVAILABLE:
        return simple_roast_fallback(year_int, artist_hours, track_hours, archetype)

    top_artist_list = [
        f"{name} ({hours:.1f} h)" for name, hours in artist_hours.items()
    ]
    top_track_list = [
        f"{name} ({hours:.1f} h)" for name, hours in track_hours.items()
    ]

    context_lines = [
        f"Year: {year_int}",
        f"Archetype: {archetype}",
        "",
        "Numeric features:",
        f"  total_hours: {feature_info['total_hours']:.2f}",
        f"  unique_artists: {feature_info['unique_artists']:.0f}",
        f"  unique_tracks: {feature_info['unique_tracks']:.0f}",
        f"  unique_albums: {feature_info['unique_albums']:.0f}",
        f"  top_artist_share: {feature_info['top_artist_share']:.3f}",
        f"  plays_per_track: {feature_info['plays_per_track']:.2f}",
        f"  top_track_hours: {feature_info['top_track_hours']:.2f}",
        f"  diversity_ratio: {feature_info['diversity_ratio']:.3f}",
        "",
        "Summary text:",
        comment_text,
        "",
        "Top artists by hours:",
        *top_artist_list,
        "",
        "Top tracks by hours:",
        *top_track_list,
    ]
    context = "\n".join(context_lines)

    prompt = (
        "You are a playful and sarcastic music critic. "
        "Given a listener's yearly Spotify stats and a taste archetype, "
        "write a longer, funny roast of their music taste for that year. "
        "Keep it light hearted and non offensive. "
        "Use 4 to 7 sentences. "
        "Reference the archetype, emotional mood, and specific artists or tracks when it helps. "
        "Style: casual, modern, a bit dramatic, but ultimately kind.\n\n"
        f"Here are the stats and summary:\n{context}\n\n"
        "Now write the roast:"
    )

    try:
        resp = requests.post(
            LOCAL_LLM_ENDPOINT,
            json={
                "model": LOCAL_LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 220,
                    "temperature": 0.9,
                },
            },
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "").strip()
        if not text:
            return simple_roast_fallback(year_int, artist_hours, track_hours, archetype)
        return text
    except Exception as e:
        print(f"Local LLM roast failed: {e}")
        return simple_roast_fallback(year_int, artist_hours, track_hours, archetype)


def get_recommendations_for_year(
    year_top_artists: pd.Series,
    overall_artist_hours: pd.Series,
    num_recs: int = 5,
) -> List[Tuple[str, float]]:
    if overall_artist_hours.empty:
        return []

    exclude = set(year_top_artists.index)
    recs: List[Tuple[str, float]] = []

    for artist, hours in overall_artist_hours.items():
        if artist in exclude:
            continue
        recs.append((artist, float(hours)))
        if len(recs) >= num_recs:
            break

    return recs


def generate_yearly_summary_images(
    df: pd.DataFrame,
    output_folder: str,
    overall_artist_hours: pd.Series,
):
    df_years = df[df["year"].notna()].copy()
    if df_years.empty:
        print("No valid year data available for yearly summary images.")
        return

    years = sorted(df_years["year"].dropna().unique())
    os.makedirs(output_folder, exist_ok=True)

    for year in years:
        year_int = int(year)
        df_y = df_years[df_years["year"] == year_int]
        if df_y.empty:
            continue

        artist_ms = (
            df_y.groupby("master_metadata_album_artist_name")["ms_played"]
            .sum()
            .sort_values(ascending=False)
            .head(YEARLY_TOP_N)
        )
        artist_hours = ms_to_hours(artist_ms)

        track_ms = (
            df_y.groupby("master_metadata_track_name")["ms_played"]
            .sum()
            .sort_values(ascending=False)
            .head(YEARLY_TOP_N)
        )
        track_hours = ms_to_hours(track_ms)

        album_ms = (
            df_y.groupby("master_metadata_album_album_name")["ms_played"]
            .sum()
            .sort_values(ascending=False)
            .head(YEARLY_TOP_N)
        )
        album_hours = ms_to_hours(album_ms)

        comment_text = build_yearly_comment(df_y, artist_hours, track_hours)

        features, feature_info = compute_year_features(df_y, artist_hours, track_hours)
        archetype = classify_taste_archetype(features)

        comment_with_arch = comment_text + f"\nOverall taste archetype: {archetype}."

        roast_text = local_llm_roast(
            year_int,
            artist_hours,
            track_hours,
            comment_with_arch,
            archetype,
            feature_info,
        )

        recs = get_recommendations_for_year(artist_hours, overall_artist_hours, num_recs=5)
        if recs:
            rec_lines = []
            for idx, (artist, hours) in enumerate(recs, start=1):
                rec_lines.append(f"{idx}. {artist} (about {hours:.1f} hours overall)")
            rec_text = "\n".join(rec_lines)
        else:
            rec_text = "No recommendations available."

        combined_text = (
            "Summary:\n"
            + comment_with_arch
            + "\n\nRoast:\n"
            + roast_text
            + "\n\nRecommendations (based on your overall listening):\n"
            + rec_text
        )

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, height_ratios=[2, 3])

        ax_artists = fig.add_subplot(gs[0, 0])
        ax_tracks = fig.add_subplot(gs[0, 1])
        ax_albums = fig.add_subplot(gs[0, 2])
        ax_text = fig.add_subplot(gs[1, :])

        if not artist_hours.empty:
            series_sorted = artist_hours.sort_values(ascending=True)
            ax_artists.barh(series_sorted.index, series_sorted.values, color=BAR_COLOR)
        ax_artists.set_title("Top artists")
        ax_artists.set_xlabel("Hours")
        ax_artists.grid(axis="x", linestyle=":", alpha=0.3)
        ax_artists.tick_params(axis="y", labelsize=7)

        if not track_hours.empty:
            series_sorted = track_hours.sort_values(ascending=True)
            ax_tracks.barh(series_sorted.index, series_sorted.values, color=BAR_COLOR)
        ax_tracks.set_title("Top tracks")
        ax_tracks.set_xlabel("Hours")
        ax_tracks.grid(axis="x", linestyle=":", alpha=0.3)
        ax_tracks.tick_params(axis="y", labelsize=7)

        if not album_hours.empty:
            series_sorted = album_hours.sort_values(ascending=True)
            ax_albums.barh(series_sorted.index, series_sorted.values, color=BAR_COLOR)
        ax_albums.set_title("Top albums")
        ax_albums.set_xlabel("Hours")
        ax_albums.grid(axis="x", linestyle=":", alpha=0.3)
        ax_albums.tick_params(axis="y", labelsize=7)

        ax_text.axis("off")
        ax_text.text(
            0.01,
            0.99,
            combined_text,
            ha="left",
            va="top",
            wrap=True,
            fontsize=8,
            bbox=dict(boxstyle="round", alpha=0.08),
        )

        fig.suptitle(f"Listening profile {year_int}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        filename = os.path.join(output_folder, f"year_{year_int}_summary.png")
        plt.savefig(filename, dpi=200)
        plt.close()
        print(f"Saved yearly summary image: {filename}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate charts and yearly roasts from Spotify extended streaming history."
    )
    parser.add_argument(
        "--data-folder",
        default=DEFAULT_DATA_FOLDER,
        help="Folder containing Spotify streaming history JSON files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_folder = args.data_folder

    print(f"Loading Spotify streaming history from: {data_folder}")
    df = load_streaming_history(data_folder)

    print(f"Total rows loaded: {len(df)}")

    df_filtered = filter_valid_music(df)
    print(f"Rows after filtering by valid music and duration: {len(df_filtered)}")

    output_folder = os.path.join(data_folder, "charts")

    overall_artist_ms = (
        df_filtered.groupby("master_metadata_album_artist_name")["ms_played"]
        .sum()
        .sort_values(ascending=False)
    )
    overall_artist_hours = ms_to_hours(overall_artist_ms)

    print("Generating overall charts...")
    generate_insights(df_filtered, output_folder)

    print("Generating multi year charts...")
    generate_yearly_charts(df_filtered, output_folder)

    print("Generating yearly summary images...")
    generate_yearly_summary_images(df_filtered, output_folder, overall_artist_hours)

    print("Generating yearly insights in terminal...")
    generate_yearly_terminal_insights(df_filtered)

    print()
    print(
        "Done. Charts and yearly summary PNGs are in the 'charts' subfolder inside your data folder."
    )


if __name__ == "__main__":
    main()
