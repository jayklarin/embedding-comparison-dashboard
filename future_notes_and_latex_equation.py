# ============================================
# 🧠 Embedding Comparison Dashboard (Final Test Run v1.0)
# ============================================
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# --- Page config ---
st.set_page_config(
    page_title="Embedding Comparison Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Load GloVe subset ---
@st.cache_data
def load_glove(file_path="glove_top10k_100d.txt"):
    embeddings = {}
    with open(file_path, "r", encoding="utf8") as f:
        for line in f:
            parts = line.split()
            word, vec = parts[0], np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vec
    return embeddings

embeddings = load_glove()

# --- Utility functions ---
def vector_op(eq):
    tokens = eq.lower().split()
    vec = np.zeros(100)
    op = "+"
    for token in tokens:
        if token in {"+", "-", "*", "/"}:
            op = token
        elif token in embeddings:
            if op == "+": vec += embeddings[token]
            elif op == "-": vec -= embeddings[token]
            elif op == "*": vec *= embeddings[token]
            elif op == "/": vec /= np.where(embeddings[token] != 0, embeddings[token], 1)
    return vec

def top_k_similar(vec, k=10):
    words, sims = [], []
    for w, v in embeddings.items():
        sim = np.dot(vec, v) / (np.linalg.norm(vec)*np.linalg.norm(v))
        words.append(w)
        sims.append(sim)
    df = pd.DataFrame({"Word": words, "Similarity": sims})
    df = df.sort_values("Similarity", ascending=False).head(k)
    return df

# --- Header ---
st.title("🧠 Embedding Comparison Dashboard")

# --- Instructions block ---
with st.expander("📘 Instructions (click to expand)"):
    st.markdown("""
### 🧠 Overview
This dashboard explores **semantic relationships between words** using pre-trained [GloVe](https://nlp.stanford.edu/projects/glove/) embeddings.  
Each word is represented as a **100-dimensional vector**, and vector arithmetic reveals how meanings shift in embedding space.

---

### ✏️ How to Use
1. Enter up to **four equations** — for example:  
   `king - man + woman` or `love - hate`
2. Choose how many **Top-K similar words** to display.  
3. The dashboard will show:
   - **Top-K Semantic Matches** (bar chart)
   - **Semantic Space (PCA Projection)**
   - **Cosine Similarity Matrix**
   - **Detailed 100-Dimensional Word Vectors**
   - **Aggregate Statistics Table**  
     *(mean, median, variance, std. dev., min, max)*

---

### ⚙️ Technical Details
Cosine similarity measures how close two words are in meaning:
""")
    st.latex(r"\text{cosine\_similarity}(A, B) = \frac{A \cdot B}{|A| \, |B|}")
    st.markdown("""
Principal Component Analysis (PCA) reduces high-dimensional vectors to 2-D for easier visualization while preserving the main variance in meaning.

Supported vector operations include:
- Addition (`+`)
- Subtraction (`-`)
- Multiplication (`*`)
- Division (`/`)

---

### 🧩 Notes
- Uses **GloVe 6B (100d)** embedding subset.
- Processes **top 10 000 alphabetic words**.
- Results are **cached** for faster updates.
- Layout automatically scales for side-by-side comparisons.

Enjoy exploring the geometry of language! 🌐
""")

# --- Input controls ---
cols = st.columns(4)
eqs = [c.text_input(f"Equation {i+1}") for i, c in enumerate(cols)]
k = st.slider("Select Top-K Similar Words", 5, 20, 10)

# --- Display Top-K results ---
st.markdown("## 📊 Top-K Semantic Matches per Equation")

non_empty_eqs = [e for e in eqs if e.strip()]
vectors, labels = [], []

if non_empty_eqs:
    cols = st.columns(len(non_empty_eqs))
    for i, eq in enumerate(non_empty_eqs):
        vec = vector_op(eq)
        df = top_k_similar(vec, k)
        fig = px.bar(df, x="Similarity", y="Word", orientation="h",
                     color_discrete_sequence=["#60A5FA"], text="Similarity")
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(yaxis=dict(autorange="reversed"), height=300)
        cols[i].plotly_chart(fig, use_container_width=True)
        vectors.append(vec)
        labels.append(eq)
else:
    st.info("👆 Enter at least one valid equation to view results.")

# --- Detailed tables ---
st.markdown("## 📚 Detailed Tables (Vectors & Stats)")
if vectors:
    for label, vec in zip(labels, vectors):
        df = pd.DataFrame([embeddings[w] for w in top_k_similar(vec, k)["Word"]],
                          index=top_k_similar(vec, k)["Word"]).T
        st.markdown(f"**{label} — Top-{k} Word Vectors (100 dims)**")
        st.dataframe(df, height=350)

    # --- Aggregate statistics ---
    all_vecs = np.array(vectors)
    stats = pd.DataFrame({
        "Mean": np.mean(all_vecs, axis=0),
        "Median": np.median(all_vecs, axis=0),
        "Variance": np.var(all_vecs, axis=0),
        "Std Dev": np.std(all_vecs, axis=0),
        "Min": np.min(all_vecs, axis=0),
        "Max": np.max(all_vecs, axis=0),
    })
    st.markdown("### 📈 Aggregate Statistics (per dimension)")
    st.dataframe(stats, height=300)

# --- PCA projection ---
if len(vectors) >= 2:
    st.markdown("## 🧭 Semantic Space (2-D PCA Projection)")
    pca = PCA(n_components=2)
    pts = pca.fit_transform(np.vstack(vectors))
    fig = go.Figure()
    for i, lbl in enumerate(labels):
        fig.add_trace(go.Scatter(
            x=[pts[i,0]], y=[pts[i,1]], mode="markers+text",
            name=lbl, text=lbl, textposition="top center",
            marker=dict(size=20, line=dict(width=2, color="black"))
        ))
    fig.update_layout(height=500, width=900, xaxis_title="x", yaxis_title="y")
    st.plotly_chart(fig, use_container_width=True)

# --- Cosine similarity matrix ---
if len(vectors) >= 2:
    st.markdown("## 🔢 Cosine Similarity Matrix")
    sim = cosine_similarity(vectors)
    fig = px.imshow(sim, text_auto=".2f", color_continuous_scale="RdBu_r",
                    x=labels, y=labels, aspect="auto")
    fig.update_layout(height=500, width=900)
    st.plotly_chart(fig, use_container_width=True)

# --- Things to Fix / Next Steps ---
with st.expander("🛠️ Things to Fix / Next Steps"):
    st.markdown("""
#### ⚙️ Technical Fixes (v1.0 polish)
1. **Top-K sorting** – verify descending order remains consistent across local and deployed environments.  
2. **Responsive chart scaling** – adjust PCA and correlation matrix heights for smaller screens.  
3. **Instruction rendering** – ensure Markdown + LaTeX render cleanly post-deployment.  
4. **Dynamic columns** – fixed `StreamlitInvalidColumnSpecError` (requires at least one non-empty equation).  
5. **Aggregate stats table** – validate min/max computations against sample vectors.

---

#### 🚀 Planned Enhancements (v1.1 ideas)
1. **🧩 Dimension Correlation Map**  
   - Compare 100 embedding dimensions across top words from each equation.  
   - Compute per-dimension Pearson or cosine correlation.  
   - Visualize as a **heatmap** or **parallel coordinates plot** (Plotly).  

2. **⚖️ Weighted Operations**  
   - Allow syntax like: `0.3 * love + 0.7 * hate` for linear blends of semantic direction.  

3. **🧠 Advanced Arithmetic**  
   - Add exponentiation (`^`), root (`√`), or normalization (`|x|`) for experimental analysis.  

4. **🧾 Metadata Panel**  
   - Show dataset source (e.g., GloVe 6B, 100d), token count, and processing notes.  

5. **🎨 Style Refinement**  
   - Compact layout toggle for smaller devices.  
   - Optional light/dark theme selector.  

---

_This dashboard is now feature-complete for v1.0, with groundwork for exploratory dimension analytics in v1.1._
""")
