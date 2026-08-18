import os
import io
import time
import pandas as pd
import zstandard as zstd


def load_pgn_zst(file_path="lichess_db_standard_rated_2015-08.pgn", max_games=100000):
    """
    Stream-reads raw PGN dataset compressed with Zstandard (.pgn.zst) or uncompressed PGN (.pgn).
    Parses required fields memory-efficiently without loading the whole file into RAM.
    """
    if not os.path.exists(file_path):
        alt_paths = ["data/lichess_db_standard_rated_2015-08.pgn", "data/lichess_db_standard_rated_2015-08.pgn.zst"]
        found = False
        for alt in alt_paths:
            if os.path.exists(alt):
                file_path = alt
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Dataset file not found at {file_path} or alternative paths.")

    print(f"[*] Parsing dataset from: {file_path} (max games limit: {max_games if max_games else 'ALL'})")
    start_time = time.time()

    games = []
    current_game = {}
    moves_lines = []

    is_zst = file_path.endswith(".zst")

    def open_stream(path):
        if is_zst:
            fh = open(path, "rb")
            dctx = zstd.ZstdDecompressor()
            reader = dctx.stream_reader(fh)
            return fh, io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
        else:
            fh = open(path, "r", encoding="utf-8", errors="ignore")
            return fh, fh

    raw_fh, text_stream = open_stream(file_path)

    try:
        for line in text_stream:
            line_str = line.strip()
            if not line_str:
                continue

            if line_str.startswith("["):
                if moves_lines and current_game:
                    current_game["Moves"] = " ".join(moves_lines)
                    games.append(current_game)
                    current_game = {}
                    moves_lines = []

                    if max_games and len(games) >= max_games:
                        break

                space_idx = line_str.find(" ")
                if space_idx != -1 and line_str.endswith("]"):
                    key = line_str[1:space_idx]
                    val = line_str[space_idx + 2 : -2]
                    current_game[key] = val
            else:
                moves_lines.append(line_str)

        if moves_lines and current_game and (not max_games or len(games) < max_games):
            current_game["Moves"] = " ".join(moves_lines)
            games.append(current_game)
    finally:
        text_stream.close()
        if is_zst:
            raw_fh.close()

    elapsed = time.time() - start_time
    print(f"[+] Parsed {len(games):,} games successfully in {elapsed:.2f} seconds.")

    df = pd.DataFrame(games)

    # Ensure required columns exist
    required_cols = [
        "White", "Black", "WhiteElo", "BlackElo", "Result",
        "ECO", "Opening", "TimeControl", "Termination", "Moves", "Event"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    return df[required_cols]


def check_dataset_stats(df):
    """
    Performs data verification and exploratory statistical checks on the dataset.
    """
    print("\n" + "=" * 60)
    print("               DATASET EXPLORATORY STATISTICS")
    print("=" * 60)
    print(f"Total parsed games       : {len(df):,}")

    print("\n--- Missing Data (NaN / None) ---")
    missing = df.isnull().sum()
    for col, count in missing.items():
        pct = (count / len(df)) * 100 if len(df) > 0 else 0
        print(f"  {col:<15}: {count:>8,} missing ({pct:.2f}%)")

    print("\n--- Result Distribution ---")
    res_dist = df["Result"].value_counts(dropna=False)
    for res, count in res_dist.items():
        pct = (count / len(df)) * 100 if len(df) > 0 else 0
        print(f"  {str(res):<15}: {count:>8,} games ({pct:.2f}%)")

    print("\n--- Elo Rating Statistics ---")
    w_elo = pd.to_numeric(df["WhiteElo"], errors="coerce")
    b_elo = pd.to_numeric(df["BlackElo"], errors="coerce")
    print(f"  White Elo -> Min: {w_elo.min()}, Max: {w_elo.max()}, Mean: {w_elo.mean():.1f}, Median: {w_elo.median():.1f}")
    print(f"  Black Elo -> Min: {b_elo.min()}, Max: {b_elo.max()}, Mean: {b_elo.mean():.1f}, Median: {b_elo.median():.1f}")

    print("\n--- Top 10 Most Common Openings ---")
    top_openings = df["Opening"].value_counts().head(10)
    for op, count in top_openings.items():
        pct = (count / len(df)) * 100 if len(df) > 0 else 0
        print(f"  {str(op):<45}: {count:>6,} ({pct:.2f}%)")

    print("=" * 60 + "\n")


def prepare_and_cache_dataset(input_zst="lichess_db_standard_rated_2015-08.pgn",
                               output_csv="data/processed_games.csv",
                               max_games=100000,
                               force_reparse=False):
    """
    Parses raw dataset and saves intermediate dataset to CSV to speed up subsequent runs.
    """
    if os.path.exists(output_csv) and not force_reparse:
        print(f"[*] Found cached dataset at '{output_csv}'. Loading...")
        df = pd.read_csv(output_csv, dtype=str)
        print(f"[+] Loaded {len(df):,} games from cache.")
        check_dataset_stats(df)
        return df

    df = load_pgn_zst(input_zst, max_games=max_games)
    check_dataset_stats(df)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[+] Saved intermediate dataset to '{output_csv}'.")

    return df


if __name__ == "__main__":
    df = prepare_and_cache_dataset(max_games=10000)
