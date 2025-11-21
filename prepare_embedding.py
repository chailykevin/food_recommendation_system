# prepare_embeddings.py
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# 1. Load dan gabung semua CSV
data_folder = Path("dataset")

all_dfs = []
for csv_file in data_folder.glob("*.csv"):
    df = pd.read_csv(csv_file)
    all_dfs.append(df)

data = pd.concat(all_dfs, ignore_index=True)

# keep only needed columns
data = data[["Title", "Ingredients", "Steps"]]

# 2. Buat kolom text gabungan
def combine_text(row):
    title = str(row["Title"])
    ingredients = str(row["Ingredients"])
    steps = str(row["Steps"])
    return f"Title: {title}. Ingredients: {ingredients}. Steps: {steps}"

data["text"] = data.apply(combine_text, axis=1)

# optional: lowercase untuk pencarian judul lebih enak
data["Title_lower"] = data["Title"].str.lower()

# 3. Load Sentence-BERT dan encode semua resep
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)

embeddings = model.encode(
    data["text"].tolist(),
    show_progress_bar=True
)

# 4. Simpan data + embeddings ke file (ini “model yang sudah disiapkan”)
data.to_pickle("recipes_data.pkl")             # DataFrame
np.save("recipes_embeddings.npy", embeddings)  # numpy array

print("Selesai: recipes_data.pkl dan recipes_embeddings.npy dibuat.")
