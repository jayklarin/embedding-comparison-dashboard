import os, re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

st.set_page_config(page_title="Embedding Comparison Dashboard", layout="wide")
PRIMARY = "#6AA6FF"

st.markdown("<h1>🧠 Embedding Comparison Dashboard</h1>", unsafe_allow_html=True)

with st.expander("📘 Instructions (click to collapse)", expanded=True):
    st.markdown("""
        **Usage:**
        - Enter up to four equations (e.g., `love - hate`, `king - man + woman`, `love * hate`)
        - Press <kbd>Tab</kbd> or <kbd>Return</kbd> to update results
        - Scroll through 100-dimension vector tables and per-dimension stats
        - Compare equations across semantic, PCA, and cosine similarity maps
    """)

VEC_FILE = "glove_top10k_100d.txt"

@st.cache_data(show_spinner=True)
def load_vectors(path):
    words, vecs = [], []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 101 and re.fullmatch(r"[a-zA-Z]+", parts[0]):
                words.append(parts[0].lower())
                vecs.append(np.array(parts[1:], dtype=np.float32))
    mat = np.vstack(vecs)
    return {"words": words, "word2idx": {w:i for i,w in enumerate(words)}, "matrix": mat}

emb = load_vectors(VEC_FILE)

def get_vec(w):
    i = emb["word2idx"].get(w.lower())
    return emb["matrix"][i] if i is not None else None

def unit(v):
    n = np.linalg.norm(v)
    return v / n if n != 0 else v

def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na*nb != 0 else 0.0

def equation_vector(expr):
    tokens = expr.strip().split()
    if not tokens: return None, []
    v = get_vec(tokens[0]); used = [tokens[0]]
    if v is None: return None, used
    i = 1
    while i + 1 < len(tokens):
        op, t = tokens[i], tokens[i+1]; vt = get_vec(t); used.append(t)
        if vt is None: return None, used
        if op == "+": v = v + vt
        elif op == "-": v = v - vt
        elif op == "*": v = v * vt
        elif op == "/": v = v / np.where(vt == 0, 1e-8, vt)
        i += 2
    return v.astype(np.float32), used

def top_k_similar(target_vec, k, exclude):
    A = emb["matrix"]; tv = unit(target_vec)
    AV = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    sims = AV @ tv
    mask = np.ones(len(emb["words"]), bool)
    for w in exclude:
        idx = emb["word2idx"].get(w.lower())
        if idx is not None: mask[idx] = False
    sims_masked = np.where(mask, sims, -np.inf)
    top_idx = np.argpartition(-sims_masked, np.arange(k))[:k]
    top_idx = top_idx[np.argsort(-sims_masked[top_idx])]
    words = [emb["words"][i] for i in top_idx]; vals = sims[top_idx]
    return words, vals, top_idx

def dims_df(idxs):
    vecs = emb["matrix"][idxs]
    cols = [f"dim_{i+1}" for i in range(vecs.shape[1])]
    return pd.DataFrame(vecs, columns=cols, index=[emb["words"][i] for i in idxs])

def stats_df(idxs):
    vecs = emb["matrix"][idxs]
    return pd.DataFrame({
        "mean": vecs.mean(axis=0),
        "median": np.median(vecs, axis=0),
        "std": vecs.std(axis=0),
        "var": vecs.var(axis=0)
    }, index=[f"dim_{i+1}" for i in range(vecs.shape[1])])

cols = st.columns(4)
eqs = [cols[0].text_input("Equation 1","love - hate"),
       cols[1].text_input("Equation 2","love / hate"),
       cols[2].text_input("Equation 3","hate / love"),
       cols[3].text_input("Equation 4","love * hate")]
top_k = st.slider("Select Top-K Similar Words", 5, 25, 10)
min_rows = max(15, top_k)

eq_vecs, used_tokens = [], []
for e in eqs:
    v, used = equation_vector(e)
    eq_vecs.append(v); used_tokens.append(set(used))

# ---- Top-K Horizontal Bars ----
st.subheader("📊 Top-K Semantic Matches per Equation")
bar_cols = st.columns(4)
top_indices = []

for i, (col, v, used) in enumerate(zip(bar_cols, eq_vecs, used_tokens)):
    with col:
        if v is None:
            st.warning(f"❌ Could not parse `{eqs[i]}`")
            top_indices.append([]); continue
        words, vals, idxs = top_k_similar(v, top_k, used)
        top_indices.append(list(idxs))
        fig, ax = plt.subplots(figsize=(4.6,3.0), dpi=150, facecolor="white")
        ax.set_facecolor("white")
        ax.barh(range(len(words)), vals[::-1], color=PRIMARY, align="center", height=0.6)
        ax.set_yticks(range(len(words))); ax.set_yticklabels(words[::-1])
        ax.invert_yaxis()
        for j,vv in enumerate(vals[::-1]):
            ax.text(vv + 0.02, j, f"{vv:.2f}", va="center", fontsize=9)
        ax.set_xlabel("Cosine Similarity"); ax.set_title(eqs[i], fontsize=13)
        ax.grid(axis="x", alpha=0.3, linestyle=":")
        st.pyplot(fig, clear_figure=True, use_container_width=False)

# ---- Data Tables ----
st.markdown("---"); st.subheader("Detailed Tables (Vectors & Stats)")
tcols = st.columns(4)
for i,col in enumerate(tcols):
    with col:
        idxs = top_indices[i][:min_rows]
        if not idxs: continue
        st.markdown(f"**{eqs[i]} — Top-{len(idxs)} Word Vectors (100 dims)**")
        st.dataframe(dims_df(idxs), height=360, use_container_width=True)
        st.markdown(f"**{eqs[i]} — Per-Dimension Summary**")
        st.dataframe(stats_df(idxs), height=360, use_container_width=True)

# ---- Semantic Space ----
st.markdown("---"); st.subheader("🧭 Semantic Space (2D PCA)")
valid = [(n,v) for n,v in zip(eqs,eq_vecs) if v is not None]
if len(valid)>=2:
    names=[n for n,_ in valid]; M=np.vstack([unit(v) for _,v in valid])
    p2=PCA(n_components=2).fit_transform(M)
    fig,ax=plt.subplots(figsize=(6.5,4.5),dpi=130)
    colors=plt.cm.tab10.colors
    for i,(name,pt) in enumerate(zip(names,p2)):
        ax.scatter(pt[0],pt[1],s=130,color=colors[i])
        ax.text(pt[0]+0.02,pt[1]+0.02,name,fontsize=13,weight="bold")
    ax.set_title("Semantic Space (2D PCA)",fontsize=15)
    ax.grid(alpha=0.3,linestyle=":")
    st.pyplot(fig,clear_figure=True)

# ---- PCA Projection ----
st.subheader("📍 PCA Projection (Equations + Top-3 Words)")
pts,lbls,cols_used=[],[],[]
for i,(name,v,idxs) in enumerate(zip(eqs,eq_vecs,top_indices)):
    if v is None: continue
    pts.append(unit(v)); lbls.append(name); cols_used.append(plt.cm.tab10.colors[i])
    for j in idxs[:3]:
        pts.append(unit(emb["matrix"][j])); lbls.append(emb["words"][j])
        cols_used.append(plt.cm.tab10.colors[i])
if len(pts)>=2:
    p=PCA(n_components=2).fit_transform(np.vstack(pts))
    fig,ax=plt.subplots(figsize=(6.5,4.5),dpi=130)
    ax.scatter(p[:,0],p[:,1],c=cols_used,s=40)
    for (x,y),lab in zip(p,lbls):
        ax.text(x+0.02,y+0.02,lab,fontsize=12)
    ax.set_title("PCA Projection (Equations + Top-3 Words)",fontsize=15)
    ax.grid(alpha=0.3,linestyle=":")
    st.pyplot(fig,clear_figure=True)

# ---- Cosine Matrix ----
st.subheader("📈 Cosine Similarity Matrix (Equations)")
valid_names=[n for n,v in zip(eqs,eq_vecs) if v is not None]
valid_vecs=[unit(v) for v in eq_vecs if v is not None]
if len(valid_vecs)>=2:
    m=len(valid_vecs); S=np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            S[i,j]=cosine(valid_vecs[i],valid_vecs[j])
    fig,ax=plt.subplots(figsize=(5.8,5),dpi=130)
    im=ax.imshow(S,vmin=0,vmax=1,cmap="coolwarm")
    ax.set_xticks(np.arange(m),labels=valid_names,rotation=25,ha="right")
    ax.set_yticks(np.arange(m),labels=valid_names)
    for i in range(m):
        for j in range(m):
            ax.text(j,i,f"{S[i,j]:.2f}",ha="center",va="center",
                    color="white" if S[i,j]>0.6 else "black")
    ax.set_title("Cosine Similarity Matrix",fontsize=15)
    fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    st.pyplot(fig,clear_figure=True)

st.caption(f"Loaded {len(emb['words']):,} words from `{VEC_FILE}` — 100 dimensions each.")
