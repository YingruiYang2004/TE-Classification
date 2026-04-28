"""H10: Per-superfamily L2 distance compactness from contrastive_embeddings.csv"""
import pandas as pd, numpy as np
from itertools import combinations

EMB_PATH = "data_analysis/vgp_model_clustering/contrastive_embeddings.csv"

df = pd.read_csv(EMB_PATH)
print("Columns:", df.columns.tolist()[:10])
print("Shape:", df.shape)
print(df.head(3))
