import os
import re
import pandas as pd
import numpy as np


def clean_moves(moves_str):
    """
    Sanitizes PGN moves text:
    - Removes move numbers (e.g. '1.', '2...', '12.')
    - Removes move comments/annotations ({ ... })
    - Removes game result flags ('1-0', '0-1', '1/2-1/2', '*')
    - Normalizes multiple spaces into a single space
    """
    if not isinstance(moves_str, str):
        return ""
    moves_str = re.sub(r"\{[^}]*\}", "", moves_str)
    moves_str = re.sub(r"\d+\.+\s*", "", moves_str)
    moves_str = re.sub(r"1-0|0-1|1/2-1/2|\*", "", moves_str)
    cleaned = " ".join(moves_str.split())
    return cleaned


def extract_opening_ply(opening_str):
    """
    Extracts theoretical opening ply count from standard PGN Opening tag string.
    Example: "Sicilian Defense: 4...Nf6" -> 8 plies
    Defaults to 8 if unspecified.
    """
    if not isinstance(opening_str, str) or opening_str == "?" or not opening_str:
        return 8
    match_triple = re.findall(r"(\d+)\.\.\.", opening_str)
    if match_triple:
        return int(match_triple[-1]) * 2
    match_single = re.findall(r"(\d+)\.", opening_str)
    if match_single:
        return int(match_single[-1]) * 2 - 1
    if ":" in opening_str:
        var_part = opening_str.split(":", 1)[1].strip()
        plies = len(clean_moves(var_part).split())
        if plies > 0:
            return min(max(plies, 2), 20)
    return 8


def parse_time_control(tc_str):
    """
    Parses PGN TimeControl tag like '300+0', '60+2', '180+0', '-'
    Returns (base_time_seconds, increment_seconds).
    If invalid or '-', returns (None, None).
    """
    if not isinstance(tc_str, str) or tc_str == "-" or "+" not in tc_str:
        return None, None
    try:
        parts = tc_str.split("+")
        base = int(parts[0])
        inc = int(parts[1])
        return base, inc
    except Exception:
        return None, None


def preprocess_data(df, filtered_csv_output="data/filtered_processed_games.csv"):
    """
    Performs data cleaning, feature engineering, and target encoding for 5 features:

    5 Selected Features:
    1. white_rating (int)
    2. black_rating (int)
    3. rating_diff = white_rating - black_rating (int)
    4. rated (binary: 1 = Rated, 0 = Casual)
    5. opening_ply (int: theoretical plies in opening definition)

    Timestamp & Duration Processing:
    - Calculates game duration in seconds.
    - Filters out games with duration <= 0 or duration = 0.

    Target & Class Distribution:
    - ResultEncoded: 0 = Black win ('0-1'), 1 = Draw ('1/2-1/2'), 2 = White win ('1-0')
    - Preserves natural class distribution without SMOTE oversampling to reflect true draw frequency (approx 3-5%).

    Exports:
    - Saves post-filtered dataset to a separate CSV file: 'data/filtered_processed_games.csv'

    Returns:
    - df_clean: Full cleaned DataFrame with all original & engineered columns
    - X: Feature matrix (5 features)
    - y: Target vector (0, 1, 2)
    - df_knn: Cleaned DataFrame for KNN Opening retrieval
    """
    raw_count = len(df)
    df_clean = df.copy()

    # 1. Convert Unix timestamps if available or calculate duration from TimeControl
    if "created_at" in df_clean.columns and "last_move_at" in df_clean.columns:
        df_clean["created_at_dt"] = pd.to_datetime(pd.to_numeric(df_clean["created_at"], errors="coerce"), unit="ms")
        df_clean["last_move_at_dt"] = pd.to_datetime(pd.to_numeric(df_clean["last_move_at"], errors="coerce"), unit="ms")
        df_clean["duration_seconds"] = (df_clean["last_move_at_dt"] - df_clean["created_at_dt"]).dt.total_seconds()
    else:
        # Fallback to TimeControl base_time + increment
        tc_parsed = df_clean["TimeControl"].apply(parse_time_control)
        df_clean["base_time"] = tc_parsed.apply(lambda x: x[0])
        df_clean["increment"] = tc_parsed.apply(lambda x: x[1])
        df_clean["duration_seconds"] = df_clean["base_time"].fillna(0) + df_clean["increment"].fillna(0)

    # Filter out duration = 0 or duration <= 0
    df_clean = df_clean[df_clean["duration_seconds"] > 0].copy()

    # 2. Clean & Convert ratings to numeric
    df_clean["white_rating"] = pd.to_numeric(df_clean["WhiteElo"], errors="coerce")
    df_clean["black_rating"] = pd.to_numeric(df_clean["BlackElo"], errors="coerce")
    df_clean = df_clean.dropna(subset=["white_rating", "black_rating"])
    df_clean["white_rating"] = df_clean["white_rating"].astype(int)
    df_clean["black_rating"] = df_clean["black_rating"].astype(int)
    df_clean = df_clean[(df_clean["white_rating"] >= 600) & (df_clean["white_rating"] <= 3500)]
    df_clean = df_clean[(df_clean["black_rating"] >= 600) & (df_clean["black_rating"] <= 3500)]

    # 3. Compute rating_diff (white_rating - black_rating)
    df_clean["rating_diff"] = df_clean["white_rating"] - df_clean["black_rating"]

    # 4. Compute rated (1 if 'Rated' in Event tag, else 0)
    if "Event" in df_clean.columns:
        df_clean["rated"] = df_clean["Event"].astype(str).apply(
            lambda ev: 1 if "rated" in ev.lower() else 0
        )
    else:
        df_clean["rated"] = 1

    # 5. Compute opening_ply (strictly theoretical, 0% data leakage)
    df_clean["opening_ply"] = df_clean["Opening"].apply(extract_opening_ply)

    # 6. Result encoding (0 = Black win, 1 = Draw, 2 = White win)
    result_map = {
        "0-1": 0,
        "1/2-1/2": 1,
        "1-0": 2
    }
    df_clean = df_clean[df_clean["Result"].isin(result_map.keys())].copy()
    df_clean["ResultEncoded"] = df_clean["Result"].map(result_map).astype(int)

    # 7. Clean moves text & filter out empty/invalid moves (< 5 chars)
    df_clean["CleanedMoves"] = df_clean["Moves"].apply(clean_moves)
    df_clean = df_clean[df_clean["CleanedMoves"].str.len() >= 5].copy()

    # Feature matrix X (5 selected features)
    feature_cols = [
        "white_rating",
        "black_rating",
        "rating_diff",
        "rated",
        "opening_ply"
    ]
    X = df_clean[feature_cols].copy()
    y = df_clean["ResultEncoded"].copy()

    # Preprocessed dataframe for KNN Opening retrieval
    df_knn = df_clean.copy()
    df_knn["CleanedMovesOpening"] = df_knn["CleanedMoves"].apply(
        lambda m: " ".join(m.split()[:15])
    )

    clean_count = len(df_clean)
    dropped_count = raw_count - clean_count
    drop_pct = (dropped_count / raw_count) * 100 if raw_count > 0 else 0

    # Export post-filtered dataset to a separate CSV file
    if filtered_csv_output:
        os.makedirs(os.path.dirname(filtered_csv_output), exist_ok=True)
        df_clean.to_csv(filtered_csv_output, index=False)
        print(f"[+] Saved post-filtered dataset to separate CSV: '{filtered_csv_output}'.")

    print(f"\n[+] Data Preprocessing & Cleaning Report:")
    print(f"    - Raw parsed records       : {raw_count:,}")
    print(f"    - Clean post-filtered records: {clean_count:,}")
    print(f"    - Filtered out records     : {dropped_count:,} ({drop_pct:.2f}% dropped due to duration=0/missing ratings/invalid results)")
    print(f"    - 5 Selected Features      : {list(X.columns)}")
    print(f"    - Target Class Counts (Natural Distribution) :\n{y.value_counts().to_dict()}")

    return df_clean, X, y, df_knn


if __name__ == "__main__":
    from data_loader import prepare_and_cache_dataset
    df = prepare_and_cache_dataset(max_games=5000)
    df_clean, X, y, df_knn = preprocess_data(df)
