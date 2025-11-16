import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
from collections import defaultdict

st.set_page_config(layout="wide", page_title="Adelaide Degree Overlap")

st.title("Adelaide University — Course and Degree Overlaps")

# ---------------------------
# Load data
# ---------------------------
degree_courses = pd.read_csv("degree_courses_clean.csv")
course_freq = pd.read_csv("course_frequency.csv")
degree_overlap = pd.read_csv("degree_overlap_pairs.csv")
disciplines = pd.read_csv("disciplines.csv")['discipline'].tolist()

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Filters & Settings")

max_shared = int(degree_overlap['shared_count'].max())
min_shared_val = int(degree_overlap['shared_count'].min())
default_val = min_shared_val 
min_shared = st.sidebar.number_input(
    "Show degree pairs sharing at least this many core courses",
    min_value=min_shared_val,
    max_value=max_shared,
    value=default_val
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

selected_discipline = st.selectbox(
    "Select Discipline",
    disciplines
)

# Now filter all data for selected_discipline, and display ONE set of analysis at a time


# ---------------------------
# Top Shared Core Courses
# ---------------------------
# For course_freq table
course_freq_display = course_freq.head(top_n).reset_index(drop=True)
course_freq_display.index += 1  

course_freq_display = course_freq_display.rename(columns={
    "course_code": "Course Code",
    "course_name": "Course Name",
    "degree_count": "Degree Count",
    # Add other renames as needed
})

st.header("Top Shared Core Courses")
st.dataframe(course_freq_display.style.set_table_styles([{'selector':'th.row_heading', 'props':[('display', 'none')]}]))



# ---------------------------
# Most Connected (Broad) Degrees
# ---------------------------
st.header("Most Connected Degrees")

# Track number of connected degrees and list of shared counts
degree_connections = defaultdict(int)
avg_shared_courses = defaultdict(list)

for _, row in degree_overlap.iterrows():
    if row['shared_count'] >= min_shared:
        # Count connections
        degree_connections[row['degree_a']] += 1
        degree_connections[row['degree_b']] += 1
        # Track shared courses for average calculation
        avg_shared_courses[row['degree_a']].append(row['shared_count'])
        avg_shared_courses[row['degree_b']].append(row['shared_count'])

# Build DataFrame
degree_ranking = pd.DataFrame({
    'Degree': list(degree_connections.keys()),
    'Average Shared Courses': [round(sum(avg_shared_courses[d])/len(avg_shared_courses[d]), 2)
                           for d in degree_connections],
    'Connected Degrees': [degree_connections[d] for d in degree_connections]
}).sort_values(['Average Shared Courses'], ascending=True)


#st.dataframe(degree_ranking)

degree_ranking_display = degree_ranking.reset_index(drop=True)
degree_ranking_display.index += 1
st.dataframe(
    degree_ranking_display.style
        .format({'Average Shared Courses': '{:.2f}'})
        .set_table_styles([{'selector':'th.row_heading', 'props':[('display', 'none')]}])
)

# ---------------------------
# Pairwise Degree Overlap Table
# ---------------------------
st.header("Pairwise Degree Overlap Table")

# for _, row in degree_overlap.iterrows():
#     if row['shared_count'] >= min_shared:
#         with st.expander(
#             f"{row['degree_a']} ↔ {row['degree_b']} ({row['shared_count']} shared courses)"
#         ):
#             shared_courses = row['shared_courses'].split("|")
#             for c in shared_courses:
#                 course_info_df = degree_courses[degree_courses['course_code']==c]
#                 if not course_info_df.empty:
#                     course_info = course_info_df.iloc[0]
#                     st.write(f"{c}: {course_info['course_name']} ({course_info['stream_clean']})")
#                 else:
#                     st.write(f"{c}: (Course info not found)")

pairs_filtered = degree_overlap[degree_overlap['shared_count'] >= min_shared].reset_index(drop=True)
page_size = 10
total_pages = (len(pairs_filtered) - 1) // page_size + 1

page_num = st.number_input(
    'Page number', min_value=1, max_value=total_pages, value=1, step=1, format="%d"
)

start_idx = (page_num - 1) * page_size
end_idx = start_idx + page_size

for _, row in pairs_filtered.iloc[start_idx:end_idx].iterrows():
    with st.expander(f"{row['degree_a']} ↔ {row['degree_b']} ({row['shared_count']} shared courses)"):
        shared_courses = row['shared_courses'].split("|")
        for c in shared_courses:
            st.write(c)


# ---------------------------
# Interactive Network Graph
# ---------------------------
st.header("Interactive Network (Degrees connected by shared core courses)")

G = nx.Graph()
for d in degree_courses['degree_name'].unique():
    G.add_node(d)
for _, r in degree_overlap[degree_overlap['shared_count']>=min_shared].iterrows():
    G.add_edge(r['degree_a'], r['degree_b'], weight=r['shared_count'])

net = Network(height='750px', width='100%', notebook=False)
net.from_nx(G)
net.show_buttons(filter_=['physics'])
net.save_graph('pyvis_graph.html')

HtmlFile = open('pyvis_graph.html','r', encoding='utf-8')
components.html(HtmlFile.read(), height=750)



# Heatmap of Degree Overlaps
import plotly.graph_objects as go

st.header("Clustered Heatmap of Degree Core Overlaps")

degrees = sorted(degree_courses['degree_name'].unique())
matrix = pd.DataFrame(0, index=degrees, columns=degrees)
for _, row in degree_overlap.iterrows():
    if row['shared_count'] >= min_shared:
        matrix.loc[row['degree_a'], row['degree_b']] = row['shared_count']
        matrix.loc[row['degree_b'], row['degree_a']] = row['shared_count']

from scipy.cluster.hierarchy import linkage, leaves_list
linkage_matrix = linkage(matrix, method='average', metric='euclidean')
ordered_indices = leaves_list(linkage_matrix)
ordered_degrees = [degrees[i] for i in ordered_indices]
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
        showscale=True,
    )
)

# Add grid lines between cells
shapes = []
for i in range(n+1):
    # Vertical lines
    shapes.append(dict(type="line",
                      x0=i-0.5, y0=-0.5,
                      x1=i-0.5, y1=n-0.5,
                      line=dict(color="black", width=1)))
    # Horizontal lines
    shapes.append(dict(type="line",
                      x0=-0.5, y0=i-0.5,
                      x1=n-0.5, y1=i-0.5,
                      line=dict(color="black", width=1)))

fig.update_layout(
    height=800,
    width=900,
    xaxis_title="Degree",
    yaxis_title="Degree",
    shapes=shapes,  # Overlay the grid lines
    dragmode=False
)

config = {
}

st.plotly_chart(fig, config=config)

