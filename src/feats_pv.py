import numpy as np, pandas as pd, scipy.sparse as sp
from feats import featurize
df = pd.read_parquet("data/clean_pv.parquet"); X, M = featurize(df["code"].tolist())
sp.save_npz("data/X_PrimeVul.npz", X); np.save("data/M_PrimeVul.npy", M); df.drop(columns=["code"]).to_parquet("data/meta_PrimeVul.parquet"); print("PrimeVul", X.shape, X.nnz)
