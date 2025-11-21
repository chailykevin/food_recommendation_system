# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import re

# ============================================================
# 1. Load data & embeddings (hasil dari prepare_embeddings.py)
# ============================================================

@st.cache_resource
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
# 2. Fuzzy Matching: cari judul paling mirip
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


def _format_ingredients(text):
    """Render ingredients string with '--' as line breaks and bullets."""
    items = [part.strip() for part in str(text).split("--") if part.strip()]
    if not items:
        return ""
    return "- " + "\n- ".join(items)


def _format_steps(text):
    """Render steps string with '--' as line breaks and numbering."""
    steps = [part.strip() for part in str(text).split("--") if part.strip()]
    if not steps:
        return ""
    return "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))


# ============================================================
# 3. Fungsi rekomendasi (pakai dedup judul seperti di recommend.py)
# ============================================================

def recommend_similar_foods(food_title, top_n=5):
    idx, best_match = find_best_title_index(food_title)

    if idx is None:
        return None, None

    query_title = data.iloc[idx]['Title']

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

        candidate_tokens = _ingredient_tokens(data.iloc[i].get("Ingredients", ""))
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


# ============================================================
# 4. UI dengan Streamlit
# ============================================================

st.set_page_config(
    page_title="Food Recommendation Demo",
    page_icon="🍽️",
    layout="wide"
)

st.title("Indonesian Food Recommendation System")
st.write(
    "Masukkan nama makanan (misalnya: *Rawon*, *Nasi Goreng*, dll), "
    "lalu sistem akan menampilkan beberapa rekomendasi makanan yang mirip."
)

col1, col2 = st.columns([2, 1])

with col1:
    default_example = "Rawon"
    food_query = st.text_input("Nama makanan", value=default_example)

with col2:
    top_n = st.number_input(
        "Jumlah rekomendasi",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )

if st.button("Cari rekomendasi"):

    if not food_query.strip():
        st.warning("Masukkan nama makanan terlebih dahulu.")
    else:
        with st.spinner("Menghitung kemiripan resep..."):
            query_title, recs = recommend_similar_foods(food_query, top_n=top_n)

        if query_title is None or recs is None:
            st.error(f"Tidak ditemukan judul yang cukup mirip dengan: '{food_query}'.")
        else:
            st.subheader(f"Hasil untuk: **{query_title}**")

            if len(recs) == 0:
                st.info("Tidak ada rekomendasi lain yang cukup berbeda judulnya.")
            else:
                # Tampilkan tabel ringkas
                st.write("Daftar rekomendasi:")

                df_show = pd.DataFrame([
                    {
                        "Title": r["Title"],
                        "Similarity": f"{r['Similarity']:.4f}",
                        "URL": r.get("URL", ""),
                    }
                    for r in recs
                ])
                st.dataframe(df_show, use_container_width=True)

                # Optional: detail per rekomendasi (accordion)
                st.write("Detail setiap rekomendasi:")
                for r in recs:
                    with st.expander(f"{r['Title']} (similarity: {r['Similarity']:.4f})"):
                        if r.get("URL"):
                            st.markdown(f"[Source URL]({r['URL']})")
                        st.markdown("**Ingredients:**")
                        st.markdown(_format_ingredients(r["Ingredients"]))
                        st.markdown("**Steps:**")
                        st.markdown(_format_steps(r["Steps"]))
else:
    st.info("Masukkan nama makanan lalu klik tombol 'Cari rekomendasi'.")
