# recommend.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import re


# ============================================================
# 1. Load data & embeddings (mirror app.py)
# ============================================================
def load_data_and_embeddings():
    data = pd.read_pickle("recipes_data.pkl")
    embeddings = np.load("recipes_embeddings.npy")

    if "Title_lower" not in data.columns:
        data["Title_lower"] = data["Title"].str.lower().str.strip()
    else:
        data["Title_lower"] = data["Title_lower"].str.strip()

    return data, embeddings


data, embeddings = load_data_and_embeddings()


# ============================================================
# 2. Shared helpers (matching app.py)
# ============================================================
def find_best_title_index(food_title, cutoff=0.6):
    food_title_lower = food_title.lower()
    all_titles = data["Title_lower"].tolist()

    candidates = difflib.get_close_matches(
        food_title_lower,
        all_titles,
        n=5,
        cutoff=cutoff
    )

    if not candidates:
        return None, None

    best_match = candidates[0]
    idx_list = data.index[data["Title_lower"] == best_match].tolist()
    if not idx_list:
        return None, None

    idx = idx_list[0]
    return idx, best_match


def _ingredient_tokens(text):
    """Tokenize ingredient string: lowercase, split by comma/semicolon, strip."""
    if pd.isna(text):
        return set()
    parts = []
    for chunk in str(text).replace(";", ",").split(","):
        token = chunk.strip().lower()
        if token:
            parts.append(token)
    return set(parts)


def _is_ingredient_duplicate(candidate_tokens, accepted_tokens, threshold=0.8):
    """Jaccard similarity to skip near-identical ingredient lists."""
    if not candidate_tokens:
        return False
    for tokens in accepted_tokens:
        if not tokens:
            continue
        inter = len(candidate_tokens & tokens)
        union = len(candidate_tokens | tokens)
        if union == 0:
            continue
        jaccard = inter / union
        if jaccard >= threshold:
            return True
    return False


def _title_bucket(title):
    """
    Bucket titles so near-identical variants collapse.
    For rawon/etc, use dish + detected protein; otherwise use first two tokens.
    """
    tokens = re.findall(r"[a-z0-9]+", title.lower())
    if not tokens:
        return title.lower().strip()

    proteins = {"sapi", "ayam", "kambing", "ikan", "udang", "bebek", "daging"}
    if "rawon" in tokens:
        protein = next((t for t in tokens if t in proteins), "default")
        return f"rawon-{protein}"

    return "-".join(tokens[:2])


def _matches_filters(title, ingredients, mood, protein):
    """Simple substring filters for mood/protein; blank filters are ignored."""
    title_l = str(title).lower()
    ing_l = str(ingredients).lower()

    if mood and (mood not in title_l and mood not in ing_l):
        return False
    if protein and (protein not in title_l and protein not in ing_l):
        return False
    return True


# ============================================================
# 3. Recommendation (align with app.py)
# ============================================================
def recommend_similar_foods(food_title, top_n=5, mood=None, protein=None):
    idx, best_match = find_best_title_index(food_title)

    if idx is None:
        return None, None

    query_title = data.iloc[idx]["Title"]

    query_embedding = embeddings[idx].reshape(1, -1)
    sim_scores = cosine_similarity(query_embedding, embeddings)[0]

    all_indices = np.argsort(sim_scores)[::-1]
    all_indices = [i for i in all_indices if i != idx]

    results = []
    seen_titles = set()
    seen_buckets = set()
    kept_ingredient_tokens = []

    for i in all_indices:
        title = data.iloc[i]["Title"]
        title_lower = data.iloc[i]["Title_lower"]
        if title_lower in seen_titles:
            continue

        bucket = _title_bucket(title)
        if bucket in seen_buckets:
            continue

        ingredients_text = data.iloc[i].get("Ingredients", "")
        if not _matches_filters(title, ingredients_text, mood, protein):
            continue

        candidate_tokens = _ingredient_tokens(ingredients_text)
        if _is_ingredient_duplicate(candidate_tokens, kept_ingredient_tokens):
            continue

        seen_titles.add(title_lower)
        seen_buckets.add(bucket)
        kept_ingredient_tokens.append(candidate_tokens)

        results.append({
            "Title": title,
            "Similarity": float(sim_scores[i]),
            "Ingredients": str(data.iloc[i]["Ingredients"]),
            "Steps": str(data.iloc[i]["Steps"]),
            "URL": str(data.iloc[i].get("URL", "")),
        })

        if len(results) >= top_n:
            break

    return query_title, results


# 4. Simple CLI usage mirroring app.py behavior
if __name__ == "__main__":
    query = input("Masukkan preferensi (contoh: 'ayam pedas'): ").strip().lower()
    if not query:
        print("Input kosong.")
    else:
        parts = query.split()
        protein = parts[0] if parts else ""
        mood = " ".join(parts[1:]) if len(parts) > 1 else ""

        query_phrase = " ".join([p for p in [protein, mood] if p]).strip()
        q_title, recs = recommend_similar_foods(query_phrase, top_n=5, mood=mood, protein=protein)

        if q_title is None or recs is None:
            print(f"Tidak ditemukan rekomendasi untuk: '{query_phrase}'.")
        elif not recs:
            print(f"Tidak ada rekomendasi lain yang cukup berbeda judulnya untuk '{q_title}'.")
        else:
            print(f"\nRekomendasi mirip untuk '{query_phrase}' (judul cocok: '{q_title}'):\n")
            for r in recs:
                print(f"- {r['Title']} (similarity: {r['Similarity']:.4f})")
