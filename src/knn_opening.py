import os
import re
import joblib
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from preprocessing import clean_moves


def extract_main_opening_name(opening_str):
    """
    Extracts the root family of an opening (e.g., 'Sicilian Defense: Modern Variations' -> 'Sicilian Defense').
    """
    if not isinstance(opening_str, str) or opening_str == "?":
        return "Unknown Opening"
    # Split by colon or comma to get root family name
    parts = re.split(r"[:,]", opening_str)
    return parts[0].strip()


def train_knn_opening(df_knn, K_list=[3, 5, 7, 9], model_save_path="models/knn_opening.joblib"):
    """
    Trains TF-IDF + NearestNeighbors model for opening prediction using MOVES ONLY.

    CRITICAL RULES:
    - Features used: Moves ONLY (first 15 moves to focus on opening phase).
    - Ratings are strictly NOT used for opening retrieval.
    """
    print("\n" + "=" * 60)
    print("           TRAINING KNN FOR SIMILAR OPENINGS (MOVES ONLY)")
    print("=" * 60)

    # Use first 15 moves for opening focus
    if "CleanedMovesOpening" in df_knn.columns:
        move_texts = df_knn["CleanedMovesOpening"].tolist()
    else:
        move_texts = df_knn["CleanedMoves"].apply(lambda m: " ".join(str(m).split()[:15])).tolist()
    print(f"[*] Total games in KNN index: {len(move_texts):,}")

    print("[*] Vectorizing moves with TF-IDF (word n-grams 1-4)...")
    vectorizer = TfidfVectorizer(
        token_pattern=r"\S+",
        ngram_range=(1, 4),
        min_df=2,
        sublinear_tf=True
    )
    X_moves = vectorizer.fit_transform(move_texts)
    print(f"[+] TF-IDF Matrix shape: {X_moves.shape} (Games x Features)")

    max_k = max(K_list)
    print(f"[*] Training NearestNeighbors (metric='cosine', max K={max_k})...")
    knn_model = NearestNeighbors(n_neighbors=max_k, metric="cosine", algorithm="brute")
    knn_model.fit(X_moves)

    # Store metadata DataFrame (only necessary columns to save memory)
    metadata_cols = ["White", "Black", "Opening", "ECO", "CleanedMoves"]
    df_meta = df_knn[metadata_cols].reset_index(drop=True)

    artifacts = {
        "vectorizer": vectorizer,
        "knn_model": knn_model,
        "metadata": df_meta,
        "K_list": K_list
    }

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(artifacts, model_save_path)
    print(f"[+] Saved KNN model and search index to '{model_save_path}'.")

    # Evaluate K values on sample queries
    evaluate_k_values(artifacts, df_meta, K_list=K_list)

    print("=" * 60 + "\n")
    return artifacts


def predict_opening(moves_input, K=5, model_or_path="models/knn_opening.joblib"):
    """
    Predicts the opening of a new game given a sequence of moves using KNN Cosine distance.

    Returns:
    - Dict with predicted opening, ECO, K, and list of K nearest games.
    """
    if isinstance(model_or_path, str):
        if not os.path.exists(model_or_path):
            raise FileNotFoundError(f"KNN model artifact not found at {model_or_path}. Please train it first.")
        artifacts = joblib.load(model_or_path)
    else:
        artifacts = model_or_path

    vectorizer = artifacts["vectorizer"]
    knn_model = artifacts["knn_model"]
    df_meta = artifacts["metadata"]

    cleaned_input = clean_moves(moves_input)
    if not cleaned_input:
        return {
            "error": "Empty or invalid move sequence provided.",
            "input_moves": moves_input
        }

    # Transform input string into TF-IDF feature vector
    input_vec = vectorizer.transform([cleaned_input])

    # Query KNN
    distances, indices = knn_model.kneighbors(input_vec, n_neighbors=K)

    distances = distances[0]
    indices = indices[0]

    nearest_games = []
    openings = []
    main_openings = []
    ecos = []

    for rank, (idx, dist) in enumerate(zip(indices, distances), start=1):
        row = df_meta.iloc[idx]
        op = row["Opening"]
        eco = row["ECO"]
        main_op = extract_main_opening_name(op)
        sim_pct = max(0.0, (1.0 - dist) * 100.0)

        nearest_games.append({
            "rank": rank,
            "dataset_index": int(idx),
            "white": row["White"],
            "black": row["Black"],
            "distance": float(dist),
            "similarity_percent": float(sim_pct),
            "opening": op,
            "main_opening": main_op,
            "eco": eco,
            "moves_excerpt": row["CleanedMoves"][:60] + "..." if len(row["CleanedMoves"]) > 60 else row["CleanedMoves"]
        })

        if op != "?" and op != "Unknown Opening":
            openings.append(op)
        if main_op != "Unknown Opening":
            main_openings.append(main_op)
        if eco != "???":
            ecos.append(eco)

    # Majority vote for predicted opening
    if main_openings:
        pred_main_op = Counter(main_openings).most_common(1)[0][0]
    elif openings:
        pred_main_op = Counter(openings).most_common(1)[0][0]
    else:
        pred_main_op = "Unknown Opening"

    # Exact opening variation majority vote
    if openings:
        pred_exact_op = Counter(openings).most_common(1)[0][0]
    else:
        pred_exact_op = pred_main_op

    # ECO majority vote
    if ecos:
        pred_eco = Counter(ecos).most_common(1)[0][0]
    else:
        pred_eco = "N/A"

    return {
        "input_moves_raw": moves_input,
        "input_moves_cleaned": cleaned_input,
        "predicted_opening": pred_exact_op,
        "predicted_main_opening": pred_main_op,
        "predicted_eco": pred_eco,
        "K": K,
        "nearest_games": nearest_games
    }


def evaluate_k_values(artifacts, df_meta, K_list=[3, 5, 7, 9], num_samples=5):
    """
    Evaluates KNN performance across different values of K on sample test sequences.
    """
    print("\n--- Evaluation of K values (3, 5, 7, 9) ---")

    test_queries = [
        "1. e4 c5 2. Nf3 d6 3. d4",                     # Sicilian Defense
        "1. d4 Nf6 2. c4 e6 3. Nc3 Bb4",               # Nimzo-Indian Defense
        "1. e4 e5 2. Nf3 Nc6 3. Bb5",                  # Ruy Lopez
        "1. e4 c6 2. d4 d5",                           # Caro-Kann Defense
        "1. e4 e6 2. d4 d5"                            # French Defense
    ]

    for query in test_queries:
        cleaned = clean_moves(query)
        print(f"\nTest Query: '{query}' -> Cleaned: '{cleaned}'")
        for k in K_list:
            res = predict_opening(query, K=k, model_or_path=artifacts)
            top_sim = res["nearest_games"][0]["similarity_percent"] if res["nearest_games"] else 0.0
            print(f"  K={k:<2}: Predicted Opening = '{res['predicted_opening']}' (ECO: {res['predicted_eco']}, Top Similarity: {top_sim:.1f}%)")


if __name__ == "__main__":
    from data_loader import prepare_and_cache_dataset
    from preprocessing import preprocess_data

    df = prepare_and_cache_dataset(max_games=10000)
    _, _, _, df_knn = preprocess_data(df)

    artifacts = train_knn_opening(df_knn, K_list=[3, 5, 7, 9])

    # Interactive test
    sample_input = "1. e4 c5 2. Nf3 d6 3. d4"
    result = predict_opening(sample_input, K=5, model_or_path=artifacts)
    print("\nSample Prediction Result for:", sample_input)
    print(f"Predicted Opening: {result['predicted_opening']} (ECO: {result['predicted_eco']})")
    for game in result["nearest_games"]:
        print(f"  Rank {game['rank']}: dist={game['distance']:.4f} ({game['similarity_percent']:.1f}%), Opening: {game['opening']}, ECO: {game['eco']}")
