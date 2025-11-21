# recommend.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load data & embeddings yang sudah disiapkan
data = pd.read_pickle("recipes_data.pkl")
embeddings = np.load("recipes_embeddings.npy")

# pastikan Title_lower ada; kalau belum ada di pkl, bisa bikin lagi:
if "Title_lower" not in data.columns:
    data["Title_lower"] = data["Title"].str.lower()

def recommend_similar_foods(food_title, top_n=5):
    # 2. Cari index berdasarkan judul (case-insensitive)
    food_title_lower = food_title.lower()
    matches = data.index[data["Title_lower"] == food_title_lower].tolist()
    if len(matches) == 0:
        raise ValueError(f"Food title '{food_title}' tidak ditemukan di dataset.")

    idx = matches[0]  # kalau ada duplikat, ambil yang pertama

    # 3. Ambil embedding resep query
    query_embedding = embeddings[idx].reshape(1, -1)

    # 4. Hitung cosine similarity dengan semua resep
    sim_scores = cosine_similarity(query_embedding, embeddings)[0]

    # 5. Urutkan dari similarity terbesar, dan buang dirinya sendiri
    similar_indices = np.argsort(sim_scores)[::-1]
    similar_indices = [i for i in similar_indices if i != idx]

    # 6. Ambil top_n
    top_indices = similar_indices[:top_n]

    # 7. Siapkan hasil
    results = []
    for i in top_indices:
        results.append({
            "Title": data.iloc[i]["Title"],
            "Similarity": float(sim_scores[i])
        })

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
