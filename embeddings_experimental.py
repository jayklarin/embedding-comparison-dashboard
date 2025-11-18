import os, re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

# ----------------- Page Setup -----------------
st.set_page_config(page_title="Embedding Comparison Dashboard (v1)", layout="wide")
PRIMARY = "#6AA6FF"
VEC_FILE = "glove_top10k_100d.txt"

st.markdown("<h1>🧑‍🧬 Embedding Comparison Dashboard (v1)</h1>", unsafe_allow_html=True)

# ----------------- Instructions -----------------
with st.expander("📘 Instructions (click to collapse)", expanded=True):
    st.markdown(r"""
**Purpose:** Explore relationships among word embeddings using arithmetic and cosine similarity.

**Vector arithmetic**
- Each word is represented by a 100-dimensional vector \( \vec{w} \in \mathbb{R}^{100} \)
- Example operations:
  - \( \texttt{love - hate} \Rightarrow \vec{love} - \vec{hate} \)
  - \( \texttt{king - man + woman} \Rightarrow \vec{king} - \vec{man} + \vec{woman} \)
  - \( \texttt{love * hate} \) multiplies components element-wise
  - \( \texttt{love / hate} \) divides components element-wise

**Similarity**
- Cosine similarity \( \cos(\theta)=\frac{\vec{a}\cdot\vec{b}}{\|\vec{a}\|\|\vec{b}\|} \)
- Higher values = more semantically similar

**Visuals**
- *Top-K Semantic Matches* — horizontal bars sorted **descending**
- *Top 5 Contributing Dimensions* — stacked bars showing most active dims
- *Semantic Space (PCA)* — 2D scatter showing clusters
- *Cosine Matrix* — pairwise similarity heatmaps

**Tables**
- “Summary” shows mean, var, std, min, max for top-k words
""")

# ----------------- Load Embeddings -----------------
@st.cache_data(show_spinner=True)
def load_vectors(path):
    words, vecs = [], []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) == 101 and re.fullmatch(r"[A-Za-z]+", p[0]):
                words.append(p[0].lower())
                vecs.append(np.array(p[1:], dtype=np.float32))
    M = np.vstack(vecs)
    return {"words": words, "word2idx": {w:i for i,w in enumerate(words)}, "matrix": M}

emb = load_vectors(VEC_FILE)

# ----------------- Utilities -----------------
def get_vec(w): 
    i = emb["word2idx"].get(w.lower())
    return emb["matrix"][i] if i is not None else None

def unit(v): 
    n = np.linalg.norm(v);  return v/n if n!=0 else v

def cosine(a,b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a,b)/(na*nb)) if na*nb!=0 else 0.0

def equation_vector(expr):
    tokens = expr.strip().split()
    if not tokens: return None, []
    v = get_vec(tokens[0]); used=[tokens[0]]
    if v is None: return None, used
    i=1
    while i+1<len(tokens):
        op,t = tokens[i], tokens[i+1]; vt=get_vec(t); used.append(t)
        if vt is None: return None, used
        if op=="+": v=v+vt
        elif op=="-": v=v-vt
        elif op=="*": v=v*vt
        elif op=="/": v=v/np.where(vt==0,1e-8,vt)
        i+=2
    return v.astype(np.float32), used

def top_k_similar(v, k, exclude):
    A=emb["matrix"]; tv=unit(v)
    AV=A/(np.linalg.norm(A,axis=1,keepdims=True)+1e-8)
    sims=AV@tv
    mask=np.ones(len(emb["words"]),bool)
    for w in exclude:
        idx=emb["word2idx"].get(w.lower())
        if idx is not None: mask[idx]=False
    sims=np.where(mask,sims,-np.inf)
    top_idx=np.argpartition(-sims,np.arange(min(k,len(sims))))[:k]
    top_idx=top_idx[np.argsort(-sims[top_idx])]
    words=[emb["words"][i] for i in top_idx]; vals=sims[top_idx]
    return words, vals, top_idx

def top_contributing_dims(vecs, top_n=5):
    rows = []
    for word, vec in vecs.items():
        top_dims = np.argsort(np.abs(vec))[-top_n:][::-1]
        for d in top_dims:
            rows.append({'word': word, 'dimension': d, 'magnitude': vec[d]})
    return pd.DataFrame(rows)

# ----------------- UI Inputs -----------------
cols=st.columns(4)
eqs=[cols[0].text_input("Equation 1","love - hate"),
     cols[1].text_input("Equation 2","love / hate"),
     cols[2].text_input("Equation 3","hate / love"),
     cols[3].text_input("Equation 4","love * hate")]
top_k=st.slider("Select Top-K Similar Words",5,25,10)

# ----------------- Top-K Chart -----------------
st.subheader("\U0001f4ca Top-K Semantic Matches per Equation")
top_words_dict = {}; eq_vecs=[]; used_tokens=[]; top_indices=[]
bcols=st.columns(4)
for i, (col, eq) in enumerate(zip(bcols, eqs)):
    v, used = equation_vector(eq)
    eq_vecs.append(v); used_tokens.append(set(used))
    if v is None:
        with col: st.warning(f"Couldn't parse: `{eq}`"); top_indices.append([]); continue
    words, vals, idxs = top_k_similar(v, top_k, used)
    top_words_dict[eq] = words; top_indices.append(list(idxs))
    with col:
        fig,ax=plt.subplots(figsize=(4.6,2.8),dpi=150,facecolor="white")
        ax.set_facecolor("white")
        ax.barh(range(len(words)), vals, color=PRIMARY, height=0.55)
        ax.set_yticks(range(len(words))); ax.set_yticklabels(words, fontsize=9)
        ax.invert_yaxis()
        for j,val in enumerate(vals):
            ax.text(val+0.02,j,f"{val:.2f}",va="center",fontsize=9)
        ax.set_xlabel("Cosine Similarity"); ax.set_title(eq,fontsize=12)
        ax.grid(axis="x",alpha=0.3,linestyle=":")
        st.pyplot(fig,clear_figure=True)

# ----------------- Contribution Charts -----------------
st.subheader("\U0001f7e9 Contribution Charts per Equation")
rows = [st.columns(4), st.columns(4)]
for i, eq in enumerate(eqs):
    top_words = top_words_dict[eq]; vecs = {w: emb["matrix"][emb["word2idx"][w]] for w in top_words}
    with rows[0][i]:
        st.markdown(f"**Top 5 Contributing Dimensions: `{eq}`**")
        df = top_contributing_dims(vecs)
        pivot = df.pivot(index='word', columns='dimension', values='magnitude').fillna(0)
        fig, ax = plt.subplots(figsize=(4, 3))
        pivot.plot(kind='bar', stacked=True, ax=ax, legend=False)
        ax.set_ylabel("Magnitude")
        ax.legend(title="Dim", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
        st.pyplot(fig)
    with rows[1][i]:
        st.markdown(f"**Semantic PCA Space: `{eq}`**")
        mat = np.stack([vecs[w] for w in top_words])
        X_pca = PCA(n_components=2).fit_transform(mat)
        fig, ax = plt.subplots(figsize=(4, 3))
        for j, word in enumerate(top_words):
            ax.scatter(X_pca[j, 0], X_pca[j, 1])
            ax.text(X_pca[j, 0]+0.02, X_pca[j, 1]+0.02, word, fontsize=8)
        st.pyplot(fig)

# ----------------- Dimension Summary -----------------
st.subheader("\U0001f4c8 Dimension Summary Tables per Equation")
dcols = st.columns(4)
for i, eq in enumerate(eqs):
    top_words = top_words_dict[eq]
    mat = np.stack([emb["matrix"][emb["word2idx"][w]] for w in top_words])
    df = pd.DataFrame(mat)
    stats = df.agg(['mean','var','std','min','max']).T.reset_index().rename(columns={'index': 'dim'})
    stats['dim'] = stats['dim'].astype(str)
    with dcols[i]: st.dataframe(stats)

# ----------------- Cosine Matrix per Equation -----------------
st.subheader("\U0001f4c1 Cosine Similarity Matrix per Equation")
cmcols = st.columns(4)
for i, eq in enumerate(eqs):
    words = top_words_dict[eq]
    mat = np.stack([emb["matrix"][emb["word2idx"][w]] for w in words])
    sim = cosine_similarity(mat)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        sim,
        xticklabels=words,
        yticklabels=words,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        annot_kws={"size": 6},  # 👈 smaller font
        ax=ax
    )

    ax.set_title(eq, fontsize=9)
    cmcols[i].pyplot(fig)

# ----------------- Dimension Polarity Extremes (Essence Explorer) -----------------
# st.header("🔮 Dimension Essences — Top Positive & Negative Words")

# Precompute once and cache
@st.cache_data(show_spinner=False)
def compute_dimension_extremes(words, matrix, top_k=15):
    dims = matrix.shape[1]
    result = {}

    for d in range(dims):
        col = matrix[:, d]

        # Top positive
        pos_idx = np.argsort(col)[-top_k:][::-1]
        # Top negative
        neg_idx = np.argsort(col)[:top_k]

        result[d] = {
            "top_positive": [(words[i], float(col[i])) for i in pos_idx],
            "top_negative": [(words[i], float(col[i])) for i in neg_idx],
        }

    return result

dimension_extremes = compute_dimension_extremes(
    emb["words"],
    emb["matrix"],
    top_k=15
)

# # UI
# dim = st.slider("Select embedding dimension", 0, 99, 0)

# st.subheader(f"📈 Dimension {dim} — Polarity Extremes")
# ext = dimension_extremes[dim]

# ----------------- Full Dimension Essence Table -----------------
st.header("🧭 Full Dimension Essence Table (All 100 Dimensions)")

@st.cache_data(show_spinner=False)
def compute_dimension_essence_table(words, matrix, top_k=10):
    dims = matrix.shape[1]
    rows = []

    for d in range(dims):
        col = matrix[:, d]

        # top-k positive
        pos_idx = np.argsort(col)[-top_k:][::-1]
        pos_words = [words[i] for i in pos_idx]

        # top-k negative
        neg_idx = np.argsort(col)[:top_k]
        neg_words = [words[i] for i in neg_idx]

        rows.append({
            "dim": d,
            "top_positive": ", ".join(pos_words),
            "top_negative": ", ".join(neg_words)
        })

    return pd.DataFrame(rows)

essence_df = compute_dimension_essence_table(
    emb["words"], emb["matrix"], top_k=10
)

st.dataframe(
    essence_df,
    use_container_width=True,
    hide_index=True
)
