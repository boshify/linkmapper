import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
import networkx as nx
from pyvis.network import Network
import tempfile

# Function to calculate relevance scores
def calculate_relevance_scores(df, title_tag_column):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df[title_tag_column])
    relevance_scores = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return relevance_scores

# Function to calculate minimum repeat limit needed
def calculate_minimum_repeat_limit(df, link_count):
    total_links_needed = len(df) * link_count
    total_unique_links = len(df)
    return max(1, math.ceil(total_links_needed / total_unique_links))

# Function to ensure every row has links and every link is used fairly
def ensure_no_row_without_links(df, link_usage, repeat_limit, link_count):
    all_urls = df['Full URL'].tolist()
    
    for idx, row in df.iterrows():
        if pd.isnull(row[f'Link 1 URL']):  # Check if the row has no links
            links_added = 0
            for url in all_urls:
                if link_usage[url] < repeat_limit or links_added < link_count:
                    for i in range(link_count):
                        if pd.isnull(row[f'Link {i+1} URL']):
                            df.at[idx, f'Link {i+1} URL'] = url
                            df.at[idx, f'Link {i+1} Anchor Text'] = df.loc[df['Full URL'] == url, 'Target Keyword'].values[0]
                            link_usage[url] += 1
                            links_added += 1
                            break
                if links_added == link_count:
                    break

# Function to create an interactive bubble graph based on the generated link table
def create_interactive_graph(df):
    G = nx.Graph()
    
    # Add nodes based on Target Keywords
    for idx, row in df.iterrows():
        G.add_node(row['Target Keyword'], title=row['Page Title Tag'], label=row['Target Keyword'])
    
    # Add edges based on the generated link table
    for idx, row in df.iterrows():
        for i in range(1, 6):  # Assuming a maximum of 5 links per row
            link_url = row[f'Link {i} URL']
            if pd.notnull(link_url):
                target_keyword = row[f'Link {i} Anchor Text']
                G.add_edge(row['Target Keyword'], target_keyword)
    
    # Create the pyvis network
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
    
    # Configure physics to enable force-directed layout
    net.barnes_hut()
    
    # Add nodes and edges from networkx graph to pyvis network
    net.from_nx(G)
    
    # Customize nodes and edges appearance
    for node in net.nodes:
        node['title'] = node['label']  # Ensure the label is visible without clicking
        node['label'] = node['id']  # Show the target keyword directly
        node['value'] = G.degree[node['id']]  # Node size based on degree (number of connections)
        node['labelHighlightBold'] = True  # Highlight label for better visibility
        node['font'] = {"size": 12}  # Adjust font size to ensure all labels are visible
    
    for edge in net.edges:
        edge['width'] = 2  # Set a consistent edge width
    
    # Enable node selection to highlight connections and dim unconnected nodes
    net.set_options("""
    var options = {
      nodes: {
        font: {
          size: 12,
          color: '#ffffff'
        }
      },
      edges: {
        color: {
          inherit: 'both'
        },
        smooth: false
      },
      interaction: {
        hover: true,
        tooltipDelay: 200,
        hideEdgesOnDrag: true
      },
      physics: {
        stabilization: {
          enabled: true,
          iterations: 1000
        }
      }
    }
    """)
    
    return net

# Streamlit UI
st.title("Internal Linking Mapper")

# Step 1: Upload CSV
uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Step 2: Map fields
    st.subheader("Map your fields")
    cluster_column = st.selectbox("Select the Cluster column", df.columns)
    hub_spoke_column = st.selectbox("Select the Hub Or Spoke column", df.columns)
    target_keyword_column = st.selectbox("Select the Target Keyword column", df.columns)
    title_tag_column = st.selectbox("Select the Page Title Tag column", df.columns)
    url_column = st.selectbox("Select the Full URL column", df.columns)
    
    # Step 3: Link Count Slider
    link_count = st.slider("Set Number of Internal Links per Page", min_value=1, max_value=10, value=5)
    
    # Calculate the minimum repeat limit required
    min_repeat_limit = calculate_minimum_repeat_limit(df, link_count)
    
    # Step 4:
