import pandas as pd
from pathlib import Path

# folder where you saved the CSVs
data_folder = Path("dataset")

# read all csv files and combine them
all_dfs = []
for csv_file in data_folder.glob("*.csv"):
    df = pd.read_csv(csv_file)
    all_dfs.append(df)

data = pd.concat(all_dfs, ignore_index=True)

# keep only the columns we need
data = data[["Title", "Ingredients", "Steps"]]

def combine_text(row):
    title = str(row["Title"])
    ingredients = str(row["Ingredients"])
    steps = str(row["Steps"])
    # simple combination; you can keep it like this
    return f"Title: {title}. Ingredients: {ingredients}. Steps: {steps}"

data["text"] = data.apply(combine_text, axis=1)

from sentence_transformers import SentenceTransformer

model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)

# This might take a bit but only needs to be done once (or rarely)
embeddings = model.encode(
    data["text"].tolist(),
    show_progress_bar=True
)

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def recommend_similar_foods(food_title, top_n=5):
    # 1. find the index of the food by its title
    matches = data.index[data["Title"] == food_title].tolist()
    if len(matches) == 0:
        raise ValueError(f"Food title '{food_title}' not found in dataset.")
    
    idx = matches[0]  # if duplicates, we just use the first one

    # 2. get the embedding of that food
    query_embedding = embeddings[idx].reshape(1, -1)  # shape (1, dim)

    # 3. compute cosine similarity with all embeddings
    sim_scores = cosine_similarity(query_embedding, embeddings)[0]  # shape (num_foods,)

    # 4. get indices of top_n most similar items (excluding itself)
    # sort from highest to lowest
    similar_indices = np.argsort(sim_scores)[::-1]

    # remove itself
    similar_indices = [i for i in similar_indices if i != idx]

    # take the top_n
    top_indices = similar_indices[:top_n]

    # 5. build result: title + similarity score
    results = []
    for i in top_indices:
        results.append({
            "Title": data.iloc[i]["Title"],
            "Similarity": float(sim_scores[i])
        })

    return results

recommendations = recommend_similar_foods("Nasi Goreng", top_n=5)
for r in recommendations:
    print(r["Title"], "-", r["Similarity"])
