import streamlit as st
import gensim.downloader as api
import numpy as np
import pandas as pd
import altair as alt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ==================================================
# 🧠 Embedding Comparison Dashboard (Tab-Focus Edition)
# ==================================================

st.set_page_config(
    page_title="Embedding Comparison Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Title & Intro ---
st.title("🧠 Embedding Comparison Dashboard")
st.write(
    "Type in four equations (e.g., `love - hate`, `king - man + woman`), "
    "press **Tab** or **Return** to update results, and compare them on a shared semantic map."
)

# --- Collapsible Instructions ---
with st.expander("📘 Instructions (click to collapse)", expanded=True):
    st.markdown("""
    **What is Cosine Similarity?**

    Cosine similarity measures how close two vectors point in the same direction:

    $$
    \\text{cosine\\_similarity}(A, B) = \\frac{A \\cdot B}{|A| \\, |B|}
    $$

    - **1.0 →** very similar meaning (same direction)  
    - **0.0 →** unrelated  
    - **−1.0 →** opposite meaning  

    Supported operations:
    - ➕ add vectors (`+`)
    - ➖ subtract vectors (`-`)
    - ✖️ multiply (`*`) — scales the combined meaning
    - ➗ divide (`/`) — weakens or inverses the influence of the next word
    """)

# --- Load Embeddings ---
@st.cache_resource(show_spinner=True)
def load_model():
    return api.load("glove-wiki-gigaword-100")

model = load_model()

# --- Compute Expression Safely ---
def compute_expression(expr):
    expr = expr.strip()
    tokens = expr.replace("*", " * ").replace("/", " / ").split()
    base_vec = None
    op = "+"
    for token in tokens:
        if token in ["+", "-", "*", "/"]:
            op = token
        elif token in model:
            vec = np.copy(model[token])
            if base_vec is None:
                base_vec = vec
            else:
                if op == "+": base_vec += vec
                elif op == "-": base_vec -= vec
                elif op == "*": base_vec *= vec
                elif op == "/":
                    denom = np.where(vec == 0, 1e-9, vec)
                    base_vec /= denom
    if base_vec is not None:
        base_vec = base_vec / np.linalg.norm(base_vec)
    return base_vec

# --- Equation Panels ---
cols = st.columns(4)
default_eqs = ["love - hate", "love / hate", "hate / love", "love * hate"]
colors = ["#3b82f6", "#f97316", "#22c55e", "#a855f7"]

equations, vectors, dataframes = [], [], []

for i, c in enumerate(cols):
    with c:
        eq = st.text_input(f"Equation {i+1}", default_eqs[i], key=f"eq{i+1}")
        vec = compute_expression(eq)
        if vec is not None:
            words, sims = zip(*model.most_similar(positive=[vec], topn=10))
            df = pd.DataFrame({"Word": words, "Similarity": sims})
            chart = (
                alt.Chart(df)
                .mark_bar(color=colors[i])
                .encode(x="Similarity:Q", y=alt.Y("Word:N", sort="-x"))
                .properties(height=250)
            )
            st.subheader(f"Equation {i+1}")
            st.altair_chart(chart, use_container_width=True)
            equations.append(eq)
            vectors.append(vec)
            dataframes.append(df)

# --- Shared PCA Plot ---
if len(vectors) > 1:
    st.markdown("### 📈 Shared Semantic Projection (PCA)")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(np.vstack(vectors))
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (x, y) in enumerate(coords):
        ax.scatter(x, y, color=colors[i], s=100, label=equations[i])
        ax.text(x + 0.02, y, equations[i], fontsize=10)
    ax.set_title("Semantic Space (2D PCA)")
    ax.legend()
    st.pyplot(fig)

# --- Pairwise Similarity Matrix ---
if len(vectors) > 1:
    st.markdown("### 🔢 Pairwise Equation Similarity")
    sim = cosine_similarity(vectors)
    df_sim = pd.DataFrame(sim, columns=equations, index=equations)
    st.dataframe(df_sim.style.background_gradient(cmap="viridis").format("{:.3f}"))
