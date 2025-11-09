import os, re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ----------------- Page Setup -----------------
st.set_page_config(page_title="Embedding Comparison Dashboard (v1)", layout="wide")
PRIMARY = "#6AA6FF"
VEC_FILE = "glove_top10k_100d.txt"

st.markdown("<h1>🧠 Embedding Comparison Dashboard (v1)</h1>", unsafe_allow_html=True)

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
- *Semantic Space (PCA)* — compares equations
- *PCA Projection* — each equation + its 3 nearest words
- *Cosine Matrix* — pairwise similarities between equations

**Tables**
- “Word Vectors” shows 100 dimensions per word (scrollable)
- “Summary” gives mean, median, std, variance across Top-N
""")

# ----------------- Things To Fix -----------------
with st.expander("🧩 Things to Fix / Improve (v1 roadmap)", expanded=False):
    st.markdown("""
- 🧮 **Instructions formatting** loses LaTeX on Streamlit Cloud — needs markdown/HTML hybrid fix  
- 📊 **Top-K Semantic Matches** sometimes flips to ascending on deploy; force stable sorting  
- 🖼️ **Chart sizing** — Semantic Space, PCA Projection, and Cosine Matrix should auto-fit small screens  
- 💡 Optional: Add `.streamlit/config.toml` to lock light theme consistency  
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
    top_idx=top_idx[np.argsort(-sims[top_idx])]   # DESC
    words=[emb["words"][i] for i in top_idx]; vals=sims[top_idx]
    return words, vals, top_idx

def dims_df(idxs):
    vecs=emb["matrix"][idxs]
    cols=[f"dim_{i+1}" for i in range(vecs.shape[1])]
    return pd.DataFrame(vecs,columns=cols,index=[emb["words"][i] for i in idxs])

def stats_df(idxs):
    vecs=emb["matrix"][idxs]
    return pd.DataFrame({
        "mean":vecs.mean(0),
        "median":np.median(vecs,0),
        "std":vecs.std(0),
        "var":vecs.var(0)
    }, index=[f"dim_{i+1}" for i in range(vecs.shape[1])])

# ----------------- UI Inputs -----------------
cols=st.columns(4)
eqs=[cols[0].text_input("Equation 1","love - hate"),
     cols[1].text_input("Equation 2","love / hate"),
     cols[2].text_input("Equation 3","hate / love"),
     cols[3].text_input("Equation 4","love * hate")]
top_k=st.slider("Select Top-K Similar Words",5,25,10)
min_rows=max(15,top_k)

eq_vecs,used_tokens=[],[]
for e in eqs:
    v,u=equation_vector(e)
    eq_vecs.append(v); used_tokens.append(set(u))

# ----------------- Top-K Bars -----------------
st.subheader("📊 Top-K Semantic Matches per Equation")
bcols=st.columns(4)
top_indices=[]
for i,(col,v,used) in enumerate(zip(bcols,eq_vecs,used_tokens)):
    with col:
        if v is None:
            st.warning(f"❌ Could not parse `{eqs[i]}`"); top_indices.append([]); continue
        words,vals,idxs=top_k_similar(v,top_k,used)
        # Ensure DESCENDING visual order
        fig,ax=plt.subplots(figsize=(4.6,2.8),dpi=150,facecolor="white")
        ax.set_facecolor("white")
        ax.barh(range(len(words)), vals, color=PRIMARY, height=0.55)
        ax.set_yticks(range(len(words))); ax.set_yticklabels(words, fontsize=9)
        ax.invert_yaxis()
        for j,val in enumerate(vals):
            ax.text(val+0.02,j,f"{val:.2f}",va="center",fontsize=9)
        ax.set_xlabel("Cosine Similarity"); ax.set_title(eqs[i],fontsize=12)
        ax.grid(axis="x",alpha=0.3,linestyle=":")
        st.pyplot(fig,clear_figure=True,use_container_width=False)
        top_indices.append(list(idxs))

# ----------------- Tables -----------------
st.markdown("---")
st.subheader("Detailed Tables (Vectors & Stats)")
tcols=st.columns(4)
for i,c in enumerate(tcols):
    with c:
        idxs=top_indices[i][:min_rows]
        if not idxs: continue
        st.markdown(f"**{eqs[i]} — Top-{len(idxs)} Word Vectors (100 dims)**")
        st.dataframe(dims_df(idxs),height=340,use_container_width=True)
        st.markdown(f"**{eqs[i]} — Per-Dimension Summary**")
        st.dataframe(stats_df(idxs),height=340,use_container_width=True)

# ----------------- Semantic Space -----------------
st.markdown("---"); st.subheader("🧭 Semantic Space (2D PCA)")
valid=[(n,v) for n,v in zip(eqs,eq_vecs) if v is not None]
if len(valid)>=2:
    names=[n for n,_ in valid]; M=np.vstack([unit(v) for _,v in valid])
    p2=PCA(n_components=2).fit_transform(M)
    fig,ax=plt.subplots(figsize=(5.8,3.8),dpi=130,facecolor="white")
    ax.set_facecolor("white")
    colors=plt.cm.tab10.colors
    for i,(nm,pt) in enumerate(zip(names,p2)):
        ax.scatter(pt[0],pt[1],s=120,color=colors[i])
        ax.text(pt[0]+0.02,pt[1]+0.02,nm,fontsize=12,weight="bold")
    ax.set_title("Semantic Space (2D PCA)",fontsize=14)
    ax.grid(alpha=0.3,linestyle=":")
    st.pyplot(fig,clear_figure=True)

# ----------------- PCA Projection -----------------
st.subheader("📍 PCA Projection (Equations + Top-3 Words)")
pts,lbls,cols_used=[],[],[]
for i,(nm,v,idxs) in enumerate(zip(eqs,eq_vecs,top_indices)):
    if v is None: continue
    pts.append(unit(v)); lbls.append(nm); cols_used.append(plt.cm.tab10.colors[i])
    for j in idxs[:3]:
        pts.append(unit(emb["matrix"][j])); lbls.append(emb["words"][j]); cols_used.append(plt.cm.tab10.colors[i])
if len(pts)>=2:
    p=PCA(n_components=2).fit_transform(np.vstack(pts))
    fig,ax=plt.subplots(figsize=(5.8,3.8),dpi=130,facecolor="white")
    ax.set_facecolor("white")
    ax.scatter(p[:,0],p[:,1],c=cols_used,s=40)
    for (x,y),lab in zip(p,lbls):
        ax.text(x+0.02,y+0.02,lab,fontsize=11)
    ax.set_title("PCA Projection (Equations + Top-3 Words)",fontsize=14)
    ax.grid(alpha=0.3,linestyle=":")
    st.pyplot(fig,clear_figure=True)

# ----------------- Cosine Matrix -----------------
st.subheader("📈 Cosine Similarity Matrix (Equations)")
names=[n for n,v in zip(eqs,eq_vecs) if v is not None]
vecs=[unit(v) for v in eq_vecs if v is not None]
if len(vecs)>=2:
    m=len(vecs); S=np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            S[i,j]=cosine(vecs[i],vecs[j])
    fig,ax=plt.subplots(figsize=(5.5,4.0),dpi=130,facecolor="white")
    ax.set_facecolor("white")
    im=ax.imshow(S,vmin=0,vmax=1,cmap="coolwarm")
    ax.set_xticks(np.arange(m)); ax.set_xticklabels(names,rotation=25,ha="right")
    ax.set_yticks(np.arange(m)); ax.set_yticklabels(names)
    for i in range(m):
        for j in range(m):
            ax.text(j,i,f"{S[i,j]:.2f}",ha="center",va="center",
                    color="white" if S[i,j]>0.6 else "black",fontsize=9)
    ax.set_title("Cosine Similarity Matrix",fontsize=14)
    fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    st.pyplot(fig,clear_figure=True)

st.caption(f"Loaded {len(emb['words']):,} words from `{VEC_FILE}` — 100 dimensions each.")
