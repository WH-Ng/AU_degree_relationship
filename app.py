import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", page_title="Adelaide Degree Overlap")

st.title("Adelaide University — Degree Overlap (Core Courses)")

# Load cleaned files
dc = pd.read_csv("degree_courses_clean.csv")
freq = pd.read_csv("course_frequency.csv")
overlap = pd.read_csv("degree_overlap_pairs.csv")
matrix = pd.read_csv("degree_overlap_matrix.csv", index_col=0)

# Sidebar filters
min_shared = st.sidebar.slider("Minimum shared core courses to show edge", 1, 5, 1)
top_n_courses = st.sidebar.number_input("Top N courses", 5, 50, 10)

# Page: Top courses
st.header("Top shared core courses")
st.dataframe(freq.head(top_n_courses))

# Page: Heatmap
st.header("Degree overlap heatmap")
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(matrix, ax=ax, cmap='viridis')
st.pyplot(fig)

# Page: Network (pyvis)
st.header("Degree network (interactive)")
# Filter edges by min_shared
edges = overlap[overlap['shared_count'] >= min_shared]
G = nx.Graph()
for d in matrix.columns:
    G.add_node(d)
for _,r in edges.iterrows():
    G.add_edge(r['degree_a'], r['degree_b'], weight=r['shared_count'])

net = Network(height='750px', width='100%', notebook=False)
net.from_nx(G)
net.show_buttons(filter_=['physics'])
net.save_graph('pyvis_graph.html')
HtmlFile = open('pyvis_graph.html','r', encoding='utf-8')
components.html(HtmlFile.read(), height=750)
