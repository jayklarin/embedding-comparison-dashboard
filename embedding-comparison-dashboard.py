import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Embedding Comparison Dashboard", layout="wide")

@st.cache_data
def load_glove_model():
    glove = {}
    with open("glove_top10k_100d.txt", "r", encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype="float32")
            glove[word] = vector
    return glove

glove = load_glove_model()

st.title("🧠 Embedding Comparison Dashboard (v8)")
st.caption("Using local GloVe subset: glove_top10k_100d.txt (10,000 words, 100 dimensions)")

cols = st.columns(4)
default_equations = ["love", "hate", "hate / love", "love * hate"]
equations = [c.text_input(f"Equation {i+1}", default_equations[i]) for i, c in enumerate(cols)]
top_k = st.slider("Select Top-K Similar Words", 3, 20, 10)

def find_similar_words(word, top_k=10):
    if word not in glove:
        return []
    base_vec = glove[word]
    sims = {w: cosine_similarity([base_vec], [v])[0][0] for w, v in glove.items()}
    return sorted(sims.items(), key=lambda x: x[1], reverse=True)[:top_k]

results = {eq: find_similar_words(eq.split()[0], top_k) for eq in equations}

# ==============================
# TOP-K MATCHES PER EQUATION
# ==============================
st.subheader("📊 Top-K Semantic Matches per Equation")
cols = st.columns(4)
for i, (eq, data) in enumerate(results.items()):
    df = pd.DataFrame(data, columns=["Word", "Cosine Similarity"])
    cols[i].bar_chart(df.set_index("Word"))

# ==============================
# PCA PROJECTION (Equations + Top Words)
# ==============================
st.subheader("🧭 PCA Projection (Equations + Top Words)")

all_words = []
for eq, data in results.items():
    all_words.append(eq)
    all_words.extend([w for w, _ in data])
unique_words = list(set(all_words))

X = np.array([glove[w] for w in unique_words if w in glove])
pca = PCA(n_components=2)
coords = pca.fit_transform(X)
df_pca = pd.DataFrame(coords, columns=["x", "y"], index=[w for w in unique_words if w in glove])

fig = px.scatter(
    df_pca,
    x="x",
    y="y",
    text=df_pca.index,
    title="PCA Projection (Equations + Top Words)",
)
fig.update_traces(textposition="top center")
fig.update_layout(height=600)
st.plotly_chart(fig, use_container_width=True)

# ==============================
# COSINE SIMILARITY MATRIX (EQUATIONS)
# ==============================
st.subheader("📈 Cosine Similarity Matrix (Equations Only)")

vectors = np.array([glove[w.split()[0]] for w in equations if w.split()[0] in glove])
sim_matrix = cosine_similarity(vectors)
corr_df = pd.DataFrame(sim_matrix, index=equations, columns=equations)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr_df, annot=True, cmap="coolwarm", square=True, cbar=True)
st.pyplot(fig)

# ==============================
# CORRELATION MATRIX (FULL TOP WORDS)
# ==============================
st.subheader("🧩 Correlation Matrix (Top Words Combined)")

all_vectors = np.array([glove[w] for w in unique_words if w in glove])
corr = np.corrcoef(all_vectors)
corr_df_full = pd.DataFrame(corr, index=[w for w in unique_words if w in glove], columns=[w for w in unique_words if w in glove])

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_df_full, cmap="viridis", xticklabels=False, yticklabels=False)
st.pyplot(fig)
