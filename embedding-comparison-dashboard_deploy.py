# embedding-comparison-dashboard_deploy.py
import os
import re
import math
import time
import pathlib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ------------- App & Style -------------
st.set_page_config(page_title="Embedding Comparison Dashboard", layout="wide")
PRIMARY = "#6AA6FF"

title_col1, title_col2 = st.columns([1, 7])
with title_col1:
    st.markdown("### 🧠")
with title_col2:
    st.markdown(
        "<h1 style='margin-top:-10px'>Embedding Comparison Dashboard</h1>",
        unsafe_allow_html=True,
    )
st.caption(
    "Type in four equations (e.g., "
    "<code>love  -  hate</code>, <code>king  -  man  +  woman</code>), "
    "press <b>Tab</b> or <b>Return</b> to update, and compare them on shared charts.",
    unsafe_allow_html=True,
)

# ------------- Data --------------------
VEC_FILE = "glove_top10k_100d.txt"  # compact file you created

@st.cache_data(show_spinner=True)
def load_vectors(path: str):
    if not os.path.exists(path):
        return {}, None
    words = []
    vecs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 101:
                continue
            w = parts[0]
            # keep only dictionary-ish words: letters only
            if not re.fullmatch(r"[a-zA-Z]+", w):
                continue
            v = np.array(list(map(float, parts[1:])), dtype=np.float32)
            words.append(w.lower())
            vecs.append(v)
    mat = np.vstack(vecs) if vecs else np.empty((0, 100), dtype=np.float32)
    word2idx = {w: i for i, w in enumerate(words)}
    return {"word2idx": word2idx, "matrix": mat, "words": words}, pathlib.Path(path).resolve()

emb, vec_path = load_vectors(VEC_FILE)
if not emb:
    st.error(
        f"Couldn't find `{VEC_FILE}` in the working folder.\n"
        "Place your 10k file alongside this script and rerun."
    )
    st.stop()

# ------------- Helpers -----------------
def get_vec(w: str) -> np.ndarray | None:
    i = emb["word2idx"].get(w.lower())
    return emb["matrix"][i] if i is not None else None

def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0: return v
    return v / n

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    da, db = np.linalg.norm(a), np.linalg.norm(b)
    if da == 0 or db == 0: return 0.0
    return float(np.dot(a, b) / (da * db))

def equation_vector(expr: str) -> tuple[np.ndarray | None, list[str]]:
    """
    Very simple left-to-right parser for expressions like:
    love - hate, king - man + woman, love * hate, love / hate
    Tokens must be separated by spaces.
    """
    tokens = expr.strip().split()
    used = []
    if not tokens:
        return None, used
    # first term
    v = get_vec(tokens[0])
    used.append(tokens[0])
    if v is None: return None, used
    i = 1
    while i + 1 < len(tokens):
        op = tokens[i]
        term = tokens[i + 1]
        vt = get_vec(term)
        used.append(term)
        if vt is None: 
            return None, used
        if op == "+":
            v = v + vt
        elif op == "-":
            v = v - vt
        elif op == "*":
            v = v * vt
        elif op == "/":
            # safe divide
            vt_safe = np.where(vt == 0, 1e-8, vt)
            v = v / vt_safe
        else:
            return None, used
        i += 2
    return v.astype(np.float32), used

def top_k_similar(target_vec: np.ndarray, k: int, exclude: set[str]):
    # cosine similarities against full 10k; exclude equation tokens
    A = emb["matrix"]
    tv = unit(target_vec)
    AV = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    sims = AV @ tv
    # mask exclusions
    mask = np.ones(len(emb["words"]), dtype=bool)
    for w in exclude:
        idx = emb["word2idx"].get(w.lower())
        if idx is not None: mask[idx] = False
    sims_masked = np.where(mask, sims, -np.inf)
    top_idx = np.argpartition(-sims_masked, np.arange(min(k, len(sims_masked))))[:k]
    top_idx = top_idx[np.argsort(-sims_masked[top_idx])]
    words = [emb["words"][i] for i in top_idx]
    vals = sims[top_idx]
    return words, vals, top_idx

def dims_df_from_indices(idx_list: list[int]) -> pd.DataFrame:
    vecs = emb["matrix"][idx_list]
    cols = [f"dim_{i+1}" for i in range(vecs.shape[1])]
    df = pd.DataFrame(vecs, columns=cols, index=[emb["words"][i] for i in idx_list])
    return df

def stats_df_from_indices(idx_list: list[int]) -> pd.DataFrame:
    vecs = emb["matrix"][idx_list]  # shape (N, 100)
    stats = {
        "mean":  vecs.mean(axis=0),
        "median": np.median(vecs, axis=0),
        "std":   vecs.std(axis=0, ddof=0),
        "var":   vecs.var(axis=0, ddof=0),
    }
    out = pd.DataFrame(stats, index=[f"dim_{i+1}" for i in range(vecs.shape[1])])
    return out

# ------------- Inputs (4 equations + one Top-K slider) -------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    eq1 = st.text_input("Equation 1", "love - hate")
with col2:
    eq2 = st.text_input("Equation 2", "love / hate")
with col3:
    eq3 = st.text_input("Equation 3", "hate / love")
with col4:
    eq4 = st.text_input("Equation 4", "love * hate")

k = st.slider("Select Top-K Similar Words", min_value=5, max_value=25, value=10, step=1)
table_n = max(15, k)  # “at least 15 rows” for the tables

# ------------- Compute equation vectors -------------
eq_texts = [eq1, eq2, eq3, eq4]
eq_names  = eq_texts[:]  # label charts with actual equations
eq_vecs, eq_used_tokens = [], []
for e in eq_texts:
    v, used = equation_vector(e)
    eq_vecs.append(v)
    eq_used_tokens.append(set(used))

# ------------- Top-K Bars (matplotlib), + Tables -------------
st.subheader("Top-K Semantic Matches per Equation")
bars = st.columns(4)

top_indices_per_eq = []  # store for tables & PCA(Top-3)

for i, (col, v, used) in enumerate(zip(bars, eq_vecs, eq_used_tokens)):
    with col:
        if v is None:
            st.warning(f"Could not resolve **{eq_names[i]}**")
            top_indices_per_eq.append([])
            continue
        words, vals, idxs = top_k_similar(v, k, used)
        top_indices_per_eq.append(list(idxs))

        # Bar (single color, descending)
        fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=150)
        y = np.arange(len(words))[::-1]
        ax.barh(y, vals[::-1], color=PRIMARY)
        ax.set_yticks(y, labels=words[::-1])
        ax.invert_yaxis()
        ax.set_xlabel("Similarity")
        ax.set_title(eq_names[i], pad=8, fontsize=13)
        ax.grid(axis="x", alpha=0.3, linestyle=":")
        st.pyplot(fig, clear_figure=True)

# Tables (vectors + stats) under the bars
st.markdown("---")
st.subheader("Detailed Tables (Vectors & Stats)")
tcols = st.columns(4)
for i, col in enumerate(tcols):
    with col:
        idxs = top_indices_per_eq[i][:table_n]
        if not idxs:
            st.info(f"No table for **{eq_names[i]}**")
            continue
        st.markdown(f"**{eq_names[i]} — Top-{len(idxs)} Word Vectors (100 dims)**")
        df_vecs = dims_df_from_indices(idxs)
        st.dataframe(df_vecs, height=400, use_container_width=True)

        st.markdown(f"**{eq_names[i]} — Per-Dimension Summary (mean, median, std, var)**")
        df_stats = stats_df_from_indices(idxs)
        st.dataframe(df_stats, height=400, use_container_width=True)

# ------------- Semantic Space (2D PCA) — equations only -------------
st.markdown("---")
st.subheader("Semantic Space (2D PCA)")
valid = [(n, v) for n, v in zip(eq_names, eq_vecs) if v is not None]
if len(valid) >= 2:
    labels = [n for n, _ in valid]
    M = np.vstack([unit(v) for _, v in valid])
    p = PCA(n_components=2).fit_transform(M)
    fig, ax = plt.subplots(figsize=(7, 5.6), dpi=150)
    colors = plt.cm.tab10.colors
    for i, (lab, pt) in enumerate(zip(labels, p)):
        ax.scatter(pt[0], pt[1], s=140, color=colors[i % len(colors)])
        ax.text(pt[0] + 0.02, pt[1] + 0.02, lab, fontsize=13, weight="bold")
    ax.set_title("Semantic Space (2D PCA)", fontsize=18, pad=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.2, linestyle=":")
    st.pyplot(fig, clear_figure=True)
else:
    st.info("Need at least two valid equations for PCA.")

# ------------- PCA Projection (Equations + Top-3 words per equation) -------------
st.subheader("PCA Projection (Equations + Top-3 Words)")
pts = []
lbls = []
cols = []
colors = plt.cm.tab10.colors
for i, (name, v, used, idxs) in enumerate(zip(eq_names, eq_vecs, eq_used_tokens, top_indices_per_eq)):
    if v is None: 
        continue
    pts.append(unit(v)); lbls.append(name); cols.append(colors[i % len(colors)])
    # include that equation's top-3 words
    top3 = idxs[:3] if idxs else []
    for j in top3:
        pts.append(unit(emb["matrix"][j]))
        lbls.append(emb["words"][j])
        cols.append(colors[i % len(colors)])

if len(pts) >= 2:
    P = np.vstack(pts)
    p2 = PCA(n_components=2).fit_transform(P)
    fig, ax = plt.subplots(figsize=(7, 5.6), dpi=150)
    ax.scatter(p2[:, 0], p2[:, 1], s=36, c=cols, alpha=0.9)
    for (x, y), lab in zip(p2, lbls):
        ax.text(x + 0.02, y + 0.02, lab, fontsize=12)
    ax.set_title("PCA Projection (Equations + Top-3 Words)", fontsize=18, pad=10)
    ax.grid(alpha=0.2, linestyle=":")
    st.pyplot(fig, clear_figure=True)
else:
    st.info("Need at least two points for PCA projection.")

# ------------- Pairwise Cosine Matrix (4×4) -------------
st.subheader("Cosine Similarity Matrix (Equations)")
valid_names = [n for n, v in zip(eq_names, eq_vecs) if v is not None]
valid_vecs  = [unit(v) for v in eq_vecs if v is not None]

if len(valid_vecs) >= 2:
    m = len(valid_vecs)
    S = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(m):
            S[i, j] = cosine(valid_vecs[i], valid_vecs[j])

    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=150)
    im = ax.imshow(S, vmin=0, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(m), labels=valid_names, rotation=20, ha="right")
    ax.set_yticks(np.arange(m), labels=valid_names)
    for i in range(m):
        for j in range(m):
            ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", color="white" if S[i,j] > 0.6 else "black")
    ax.set_title("Cosine Similarity Matrix", fontsize=16, pad=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig, clear_figure=True)
else:
    st.info("Need at least two valid equations for the cosine matrix.")

# ------------- Footer -------------
st.caption(
    f"Loaded **{len(emb['words']):,} words** from `{vec_path.name}` — vectors: 100 dims."
)
