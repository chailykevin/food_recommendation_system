# app.py
import streamlit as st
import pandas as pd
from recommend import (
    load_data_and_embeddings,
    recommend_similar_foods,
)

# ============================================================
# 1. Load data & embeddings (hasil dari prepare_embeddings.py)
# ============================================================

data, embeddings = st.cache_resource(load_data_and_embeddings)()


# ============================================================
# Formatting helpers remain local to the app for rendering
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
# 4. UI dengan Streamlit
# ============================================================

st.set_page_config(
    page_title="Food Recommendation Demo",
    page_icon="🍽️",
    layout="wide"
)

st.title("Indonesian Food Recommendation System")
st.write(
    "Masukkan suasana/mood (misal: *pedas*, *manis*, *goreng*) dan sumber protein "
    "(misal: *ayam*, *sapi*, *udang*). Sistem akan mencari resep paling mirip "
    "yang cocok dengan preferensi tersebut."
)

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    protein_options = ["ayam", "ikan", "kambing", "sapi", "tahu", "telur", "tempe", "udang"]
    protein = st.selectbox("Protein utama", protein_options, index=0)

with col2:
    mood = st.text_input("Rasa atau teknik masak", value="pedas")

with col3:
    top_n = st.number_input(
        "Jumlah rekomendasi",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )

if st.button("Cari rekomendasi"):

    mood = mood.strip().lower()
    protein = protein.strip().lower()

    if not mood and not protein:
        st.warning("Isi minimal salah satu: mood atau protein.")
    else:
        # Build query phrase prioritizing protein then mood to align with UI order
        query_phrase = " ".join([p for p in [protein, mood] if p]).strip()

        with st.spinner("Menghitung kemiripan resep..."):
            query_title, recs = recommend_similar_foods(
                query_phrase,
                top_n=top_n,
                mood=mood,
                protein=protein
            )

        if query_title is None or recs is None:
            st.error(
                f"Tidak ditemukan resep yang cukup mirip untuk: '{query_phrase}'. "
                "Coba variasikan mood atau protein."
            )
        else:
            # Tampilkan sesuai input user, bukan judul yang sudah dicocokkan
            st.subheader(f"Hasil untuk preferensi: **{query_phrase}**")

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
    st.info("Masukkan mood/protein lalu klik tombol 'Cari rekomendasi'.")
