import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from itertools import combinations
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, leaves_list
import plotly.graph_objects as go

# ---------------------------
# Streamlit page setup
# ---------------------------
st.set_page_config(layout="wide", page_title="Adelaide Degree Overlap")
st.title("Adelaide University — Course and Degree Overlaps")

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv("all-study-areas-degrees.csv")

# ---------------------------
# Clean and map stream
# ---------------------------
def map_stream(s):
    if pd.isna(s):
        return 'Program Core'
    s_lower = s.strip().lower()
    
    if "program core" in s_lower or s_lower.startswith("program core") or s_lower.startswith("program core selectives") or s_lower.startswith("'program core - selective'"):
        return "Program Core"
        
    if "core courses" in s_lower or s_lower.startswith("core selective courses"):
        return "Core Selectives"
        
    if "common core" in s_lower or s_lower.startswith("common core"):
        return "Common Core"
        
    return s.strip()

df['stream_clean'] = df['stream'].apply(map_stream)

# Filter to only Program Core / Core Selectives (exclude Common Core)
df_filtered = df[df['stream_clean'].isin(['Program Core','Core Selectives'])]

# Rename for consistency
acc_degrees = df_filtered[['degree','discipline','course_code','course_name','stream_clean']].rename(
    columns={'degree':'degree_name'}
)

# ---------------------------
# Sidebar controls
# ---------------------------
disciplines = sorted(acc_degrees['discipline'].unique())

st.sidebar.header("Filters & Settings")

selected_disciplines = st.sidebar.multiselect(
    "Select Discipline(s) to include",
    disciplines,
    default=disciplines
)

# Filter degrees by selected disciplines
degrees_selected = acc_degrees[acc_degrees['discipline'].isin(selected_disciplines)]

# Prepare pairwise overlaps
degree_to_courses = {d: set(degrees_selected[degrees_selected['degree_name']==d]['course_code']) 
                     for d in degrees_selected['degree_name'].unique()}

# Build pairwise overlap DataFrame
rows = []
for a,b in combinations(degree_to_courses.keys(),2):
    shared = degree_to_courses[a].intersection(degree_to_courses[b])
    shared_count = len(shared)
    if shared_count >= 1:
        rows.append({
            'degree_a': a,
            'degree_b': b,
            'shared_count': shared_count,
            'shared_courses': ", ".join(sorted(shared))
        })

degree_overlap = pd.DataFrame(rows).sort_values('shared_count', ascending=False)

# Sidebar numeric inputs
if not degree_overlap.empty:
    min_shared_val = int(degree_overlap['shared_count'].min())
    max_shared_val = int(degree_overlap['shared_count'].max())
else:
    min_shared_val = 1
    max_shared_val = 1

min_shared = st.sidebar.number_input(
    "Show degree pairs sharing at least this many core courses",
    min_value=min_shared_val,
    max_value=max_shared_val,
    value=min_shared_val
)

top_n = st.sidebar.number_input(
    "Top shared courses to display", 5, 20, 10
)

st.sidebar.markdown(
    """
**Notes:**  
- Only Program Core and Core Selectives are included.  
- Pairwise table and network only show degree pairs sharing at least the selected number of courses.  
- 'Most Connected Degrees' shows:  
  - **Connected Degrees**: number of other degrees sharing ≥ threshold courses.  
  - **Average Shared Courses**: mean number of shared courses per connected degree.
"""
)

# ---------------------------
# Top Shared Core Courses
# ---------------------------
st.header("Top Shared Core Courses")

course_freq = degrees_selected.groupby(['course_code','course_name'])['degree_name'].nunique().reset_index()
course_freq = course_freq.rename(columns={'degree_name':'degree_count'}).sort_values('degree_count', ascending=False)

course_freq_display = course_freq.head(top_n).reset_index(drop=True)
course_freq_display.index += 1
st.dataframe(course_freq_display.style.set_table_styles([{'selector':'th.row_heading', 'props':[('display', 'none')]}]))

# ---------------------------
# Most Connected Degrees
# ---------------------------
st.header("Most Connected Degrees")

degree_connections = defaultdict(int)
avg_shared_courses = defaultdict(list)

for _, row in degree_overlap.iterrows():
    if row['shared_count'] >= min_shared:
        degree_connections[row['degree_a']] += 1
        degree_connections[row['degree_b']] += 1
        avg_shared_courses[row['degree_a']].append(row['shared_count'])
        avg_shared_courses[row['degree_b']].append(row['shared_count'])

degree_ranking = pd.DataFrame({
    'Degree': list(degree_connections.keys()),
    'Average Shared Courses': [round(sum(avg_shared_courses[d])/len(avg_shared_courses[d]),2)
                               for d in degree_connections],
    'Connected Degrees': [degree_connections[d] for d in degree_connections]
}).sort_values(['Average Shared Courses'], ascending=False)

degree_ranking_display = degree_ranking.reset_index(drop=True)
degree_ranking_display.index += 1
st.dataframe(
    degree_ranking_display.style
        .format({'Average Shared Courses':'{:.2f}'})
        .set_table_styles([{'selector':'th.row_heading', 'props':[('display','none')]}])
)

# ---------------------------
# Pairwise Degree Overlap Table
# ---------------------------
st.header("Pairwise Degree Overlap Table")

pairs_filtered = degree_overlap[degree_overlap['shared_count'] >= min_shared].reset_index(drop=True)
page_size = 10
total_pages = (len(pairs_filtered)-1)//page_size + 1

page_num = st.number_input(
    'Page number', min_value=1, max_value=max(1,total_pages), value=1, step=1, format="%d"
)

start_idx = (page_num-1)*page_size
end_idx = start_idx+page_size

for _, row in pairs_filtered.iloc[start_idx:end_idx].iterrows():
    with st.expander(f"{row['degree_a']} ↔ {row['degree_b']} ({row['shared_count']} shared courses)"):
        shared_courses = row['shared_courses'].split(", ")
        for c in shared_courses:
            st.write(c)

# ---------------------------
# Interactive Network Graph
# ---------------------------
st.header("Interactive Network (Degrees connected by shared core courses)")

G = nx.Graph()
for d in degrees_selected['degree_name'].unique():
    G.add_node(d)
for _, r in degree_overlap[degree_overlap['shared_count']>=min_shared].iterrows():
    G.add_edge(r['degree_a'], r['degree_b'], weight=r['shared_count'])

net = Network(height='750px', width='100%', notebook=False)
net.from_nx(G)
net.show_buttons(filter_=['physics'])
net.save_graph('pyvis_graph.html')

HtmlFile = open('pyvis_graph.html','r', encoding='utf-8')
components.html(HtmlFile.read(), height=750)

# ---------------------------
# Clustered Heatmap
# ---------------------------
st.header("Clustered Heatmap of Degree Core Overlaps")

degrees_sorted = sorted(degrees_selected['degree_name'].unique())
matrix = pd.DataFrame(0, index=degrees_sorted, columns=degrees_sorted)

for _, row in degree_overlap.iterrows():
    if row['shared_count'] >= min_shared:
        matrix.loc[row['degree_a'], row['degree_b']] = row['shared_count']
        matrix.loc[row['degree_b'], row['degree_a']] = row['shared_count']

linkage_matrix = linkage(matrix, method='average', metric='euclidean')
ordered_indices = leaves_list(linkage_matrix)
ordered_degrees = [degrees_sorted[i] for i in ordered_indices]
matrix = matrix.loc[ordered_degrees, ordered_degrees]
n = len(matrix)

fig = go.Figure(
    data=go.Heatmap(
        z=matrix.values,
        x=matrix.columns,
        y=matrix.index,
        colorscale='Viridis',
        hoverongaps=False,
        colorbar=dict(title='Shared Core Courses'),
        text=matrix.values,
        hovertemplate='Degrees: %{y} ↔ %{x}<br>Shared courses: %{z}<extra></extra>',
        showscale=True
    )
)

# Add grid lines
shapes = []
for i in range(n+1):
    shapes.append(dict(type="line", x0=i-0.5, y0=-0.5, x1=i-0.5, y1=n-0.5, line=dict(color="black", width=1)))
    shapes.append(dict(type="line", x0=-0.5, y0=i-0.5, x1=n-0.5, y1=i-0.5, line=dict(color="black", width=1)))

fig.update_layout(height=800, width=900, xaxis_title="Degree", yaxis_title="Degree", shapes=shapes, dragmode=False)
st.plotly_chart(fig)
