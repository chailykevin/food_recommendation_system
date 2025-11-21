# recommend.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# 1. Load data & embeddings yang sudah disiapkan
data = pd.read_pickle("recipes_data.pkl")
embeddings = np.load("recipes_embeddings.npy")

# pastikan Title_lower ada; kalau belum ada di pkl, bisa bikin lagi:
if "Title_lower" not in data.columns:
    data["Title_lower"] = data["Title"].str.lower().str.strip()
else:
    data["Title_lower"] = data["Title_lower"].str.strip()

def find_best_title_index(food_title, cutoff=0.6):
    """
    Mencari judul yang paling mirip dengan input user menggunakan fuzzy matching.
    Mengembalikan index baris di DataFrame `data`.
    """
    food_title_lower = food_title.lower()
    all_titles = data["Title_lower"].tolist()

    # cari beberapa judul yang paling mirip
    candidates = difflib.get_close_matches(
        food_title_lower,
        all_titles,
        n=5,       # ambil maksimal 5 kandidat mirip
        cutoff=cutoff  # minimal kemiripan (0.0–1.0)
    )

    if not candidates:
        # tidak ada yang cukup mirip
        raise ValueError(f"Judul '{food_title}' tidak ditemukan atau terlalu berbeda dari data.")

    # pakai kandidat paling mirip
    best_match = candidates[0]

    # cari index di DataFrame untuk judul itu
    idx = data.index[data["Title_lower"] == best_match].tolist()[0]

    return idx, best_match


def recommend_similar_foods(food_title, top_n=5):
    """
    Cari top_n makanan paling mirip berdasarkan embedding + cosine similarity.
    Pakai fuzzy matching untuk menemukan judul yang paling mendekati.
    """

    # Fuzzy match judul
    idx, best_match = find_best_title_index(food_title)

    print(f"\nInput: '{food_title}' → cocok dengan judul: '{data.iloc[idx]['Title']}'\n")

    # Ambil embedding makanan query
    query_embedding = embeddings[idx].reshape(1, -1)

    # Hitung cosine similarity dengan semua resep
    sim_scores = cosine_similarity(query_embedding, embeddings)[0]

    # Urutkan dari similarity paling besar → kecil
    similar_indices = np.argsort(sim_scores)[::-1]

    # Hapus dirinya sendiri
    similar_indices = [i for i in similar_indices if i != idx]

    # Ambil Top-N sambil mengabaikan duplikat judul (case-insensitive, trim)
    results = []
    seen_titles = set()
    for i in similar_indices:
        title = data.iloc[i]["Title"]
        key = title.lower().strip()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        results.append({
            "Title": title,
            "Similarity": float(sim_scores[i])
        })
        if len(results) == top_n:
            break

    return results

# 8. Contoh pemakaian sebagai script interaktif
if __name__ == "__main__":
    try:
        query = input("Masukkan nama makanan (persis seperti di dataset): ")
        recs = recommend_similar_foods(query, top_n=5)
        print(f"\nRekomendasi mirip '{query}':\n")
        for r in recs:
            print(f"- {r['Title']} (similarity: {r['Similarity']:.4f})")
    except ValueError as e:
        print(e)
