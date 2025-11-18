import os, re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -------------------- SETUP --------------------
st.set_page_config(page_title="Embedding Comparison Dashboard", layout="wide")
PRIMARY = "#6AA6FF"
GREEN = "#4CAF50"
VEC_FILE = "glove_top10k_100d.txt"

# -------------------- TITLE --------------------
st.markdown("<h1>🧠 Embedding Comparison Dashboard</h1>", unsafe_allow_html=True)

# -------------------- INSTRUCTIONS --------------------
with st.expander("📘 Instructions (click to collapse)", expanded=False):
    st.markdown(r"""
Analyze vector arithmetic on word embeddings and explore:
- Top-K Semantic Matches  
- Most Influential Dimensions  
- PCA Projections  
- Cosine Similarity Matrix  

Vector operations:
- \( \vec{a} + \vec{b} \)
- \( \vec{a} - \vec{b} \)
- \( \vec{a} \odot \vec{b} \)
- \( \vec{a} \oslash \vec{b} \)

Cosine similarity:
\[
\cos(\theta) = \frac{\vec{a}\cdot\vec{b}}{\|\vec{a}\|\,\|\vec{b}\|}
\]
""")

# -------------------- LOAD EMBEDDINGS --------------------
@st.cache_data(show_spinner=True)
def load_vectors(path):
    words, vecs = [], []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) == 101 and re.fullmatch(r"[A-Za-z]+", p[0]):
                words.append(p[0].lower())
                vecs.append(np.array(p[1:], dtype=np.float32))
    return {
        "words": words,
        "word2idx": {w: i for i, w in enumerate(words)},
        "matrix": np.vstack(vecs)
    }

emb = load_vectors(VEC_FILE)

# -------------------- UTILS --------------------
def get_vec(w):
    idx = emb["word2idx"].get(w.lower())
    return emb["matrix"][idx] if idx is not None else None

def unit(v):
    n = np.linalg.norm(v)
    return v/n if n != 0 else v

def cosine(a,b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a,b)/(na*nb)) if na*nb != 0 else 0.0

def equation_vector(expr):
    tokens = expr.strip().split()
    if not tokens:
        return None, []
    v = get_vec(tokens[0])
    used = [tokens[0]]
    if v is None:
        return None, used

    i = 1
    while i+1 <= len(tokens)-1:
        op, t = tokens[i], tokens[i+1]
        vt = get_vec(t)
        used.append(t)
        if vt is None:
            return None, used

        if op == "+":  v = v + vt
        elif op == "-": v = v - vt
        elif op == "*": v = v * vt
        elif op == "/": v = v / np.where(vt == 0, 1e-8, vt)

        i += 2

    return v.astype(np.float32), used

def top_k_similar(v, k, exclude):
    A = emb["matrix"]
    tv = unit(v)
    AV = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    sims = AV @ tv

    mask = np.ones(len(emb["words"]), bool)
    for w in exclude:
        idx = emb["word2idx"].get(w.lower())
        if idx is not None:
            mask[idx] = False

    sims = np.where(mask, sims, -np.inf)
    top = np.argpartition(-sims, np.arange(k))[:k]
    top = top[np.argsort(-sims[top])]   # DESCENDING

    return [emb["words"][i] for i in top], sims[top], top

# -------------------- INPUTS --------------------
cols = st.columns(4)
eqs = [
    cols[0].text_input("Equation 1", "love - hate"),
    cols[1].text_input("Equation 2", "love / hate"),
    cols[2].text_input("Equation 3", "hate / love"),
    cols[3].text_input("Equation 4", "love * hate"),
]
top_k = st.slider("Select Top-K Similar Words", 5, 25, 10)

# -------------------- PARSE EQUATIONS --------------------
eq_vecs, used_tokens = [], []
for e in eqs:
    v, tok = equation_vector(e)
    eq_vecs.append(v)
    used_tokens.append(set(tok))

# -------------------- TOP-K + MOST-INFLUENTIAL CHARTS --------------------
st.subheader("📊 Top-K Semantic Matches + Most Influential Dimensions (per Equation)")

top_indices_all = []
chart_cols = st.columns(4)

for i, (col, v, used) in enumerate(zip(chart_cols, eq_vecs, used_tokens)):
    with col:
        if v is None:
            st.warning(f"Could not parse `{eqs[i]}`")
            top_indices_all.append([])
            continue

        # -------- BLUE CHART --------
        words, vals, idxs = top_k_similar(v, top_k, used)
        top_indices_all.append(idxs)

        fig, ax = plt.subplots(figsize=(4,2.4), dpi=140)
        ax.barh(range(len(words)), vals, color=PRIMARY)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(eqs[i])
        ax.set_xlabel("Cosine Similarity")
        st.pyplot(fig, clear_figure=True)

        # -------- GREEN CHART --------
        submatrix = emb["matrix"][idxs]
        max_dims = np.argmax(np.abs(submatrix), axis=1)
        counts = np.bincount(max_dims, minlength=100)

        nz_dims = np.where(counts > 0)[0]
        nz_counts = counts[nz_dims]
        order = np.argsort(-nz_counts)

        dims_sorted = nz_dims[order]
        counts_sorted = nz_counts[order]
        labels = [f"dim_{d+1}" for d in dims_sorted]

        fig2, ax2 = plt.subplots(
            figsize=(4, max(2, 0.22 * len(dims_sorted))),
            dpi=140
        )
        ax2.barh(labels, counts_sorted, color=GREEN)
        ax2.invert_yaxis()
        ax2.set_title("Most Influential Dimensions")
        ax2.set_xlabel("Count")
        st.pyplot(fig2, clear_figure=True)

# ========================================================================
# ================  NOW THE ORIGINAL APP CONTINUES BELOW  ================
# ========================================================================

st.markdown("---")
st.subheader("Detailed Tables (Vectors & Stats)")

# ---------- TABLE HELPERS ----------
def dims_df(idxs):
    vecs = emb["matrix"][idxs]
    cols = [f"dim_{i+1}" for i in range(vecs.shape[1])]
    return pd.DataFrame(vecs, columns=cols,
                        index=[emb["words"][i] for i in idxs])

def stats_df(idxs):
    vecs = emb["matrix"][idxs]
    return pd.DataFrame({
        "mean": vecs.mean(0),
        "std": vecs.std(0),
        "var": vecs.var(0),
        "min": vecs.min(0),
        "max": vecs.max(0),
    }, index=[f"dim_{i+1}" for i in range(vecs.shape[1])])

# ---------- TABLES ----------
tcols = st.columns(4)
for i, c in enumerate(tcols):
    with c:
        idxs = top_indices_all[i][:max(10, top_k)]
        if len(idxs) == 0:
            continue

        st.markdown(f"**{eqs[i]} — Word Vectors (Top {len(idxs)})**")
        st.dataframe(dims_df(idxs), height=350)

        st.markdown(f"**Per-Dimension Summary**")
        st.dataframe(stats_df(idxs), height=350)

# ---------- SEMANTIC SPACE ----------
st.markdown("---")
st.subheader("🧭 Semantic Space (2D PCA)")

valid = [(name, v) for name, v in zip(eqs, eq_vecs) if v is not None]
if len(valid) >= 2:
    names = [n for n,_ in valid]
    M = np.vstack([unit(v) for _,v in valid])
    p2 = PCA(n_components=2).fit_transform(M)

    fig, ax = plt.subplots(figsize=(5.5,3.8), dpi=140)
    colors = plt.cm.tab10.colors
    for i, (nm, pt) in enumerate(zip(names, p2)):
        ax.scatter(pt[0], pt[1], color=colors[i], s=140)
        ax.text(pt[0]+0.02, pt[1]+0.02, nm, fontsize=12)
    ax.grid(alpha=0.3, linestyle=":")
    ax.set_title("Semantic Space (2D PCA)")
    st.pyplot(fig, clear_figure=True)

# ---------- PCA PROJECTION ----------
st.subheader("📍 PCA Projection (Equations + Top-3 Words)")
pts, lbls, cols_used = [], [], []

for i, (nm, v, idxs) in enumerate(zip(eqs, eq_vecs, top_indices_all)):
    if v is None:
        continue

    pts.append(unit(v))
    lbls.append(nm)
    cols_used.append(plt.cm.tab10.colors[i])

    for j in idxs[:3]:
        pts.append(unit(emb["matrix"][j]))
        lbls.append(emb["words"][j])
        cols_used.append(plt.cm.tab10.colors[i])

if len(pts) >= 2:
    P = PCA(n_components=2).fit_transform(np.vstack(pts))
    fig, ax = plt.subplots(figsize=(5.5,3.8), dpi=140)
    ax.scatter(P[:,0], P[:,1], c=cols_used)
    for (x,y), lab in zip(P, lbls):
        ax.text(x+0.02, y+0.02, lab, fontsize=10)
    ax.grid(alpha=0.3, linestyle=":")
    ax.set_title("PCA Projection")
    st.pyplot(fig, clear_figure=True)

# ---------- COSINE SIMILARITY MATRIX ----------
st.subheader("📈 Cosine Similarity Matrix (Equations)")
valid_vecs = [(n, unit(v)) for n,v in zip(eqs, eq_vecs) if v is not None]

if len(valid_vecs) >= 2:
    names = [n for n,_ in valid_vecs]
    vecs = [v for _,v in valid_vecs]
    m = len(vecs)
    S = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            S[i,j] = cosine(vecs[i], vecs[j])

    fig, ax = plt.subplots(figsize=(5.5,4), dpi=140)
    im = ax.imshow(S, cmap="coolwarm", vmin=0, vmax=1)
    ax.set_xticks(np.arange(m))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_yticks(np.arange(m))
    ax.set_yticklabels(names)
    for i in range(m):
        for j in range(m):
            ax.text(j, i, f"{S[i,j]:.2f}",
                    ha="center", va="center",
                    color="white" if S[i,j] > 0.6 else "black",
                    fontsize=9)
    ax.set_title("Cosine Similarity Matrix")
    fig.colorbar(im, ax=ax)
    st.pyplot(fig, clear_figure=True)

# ---------- FOOTER ----------
st.caption(f"Loaded {len(emb['words']):,} words from `{VEC_FILE}` — 100 dimensions each.")
