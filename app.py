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
                if links_added was the link_count:
                    break

# Function to create an interactive bubble graph based on the generated link table
def create_interactive_graph(df):
    G = nx.Graph()
    
    # Add nodes based on Target Keywords
    for idx, row in df.iterrows():
        G.add_node(row['Target Keyword'], title=row['Page Title Tag'], label=row['Target Keyword'])
    
    # Add edges based on the generated link table
    for idx, row in df.iterrows():
        for i in range(1, 11):  # Adjust to handle up to 10 links per row (link_count max value)
            link_url_col = f'Link {i} URL'
            if link_url_col in df.columns and pd.notnull(row[link_url_col]):
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
        node['title'] = f"{node['label']} (Links: {G.degree[node['id']]})"  # Tooltip with link count
        node['label'] = node['id']  # Show the target keyword directly
        node['value'] = G.degree[node['id']]  # Node size based on degree (number of connections)
        node['labelHighlightBold'] = True  # Highlight label for better visibility
        node['font'] = {"size": 12}  # Adjust font size to ensure all labels are visible
    
    for edge in net.edges:
        edge['width'] = 2  # Set a consistent edge width
    
    # Enable node selection to highlight connections and dim unconnected nodes
    net.set_options('''
    {
      "nodes": {
        "font": {
          "size": 12,
          "color": "#ffffff"
        }
      },
      "edges": {
        "color": {
          "inherit": "both"
        },
        "smooth": false
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200,
        "hideEdgesOnDrag": true
      },
      "physics": {
        "stabilization": {
          "enabled": true,
          "iterations": 1000
        }
      }
    }
    ''')
    
    return net

# Streamlit UI
st.title("Internal Linking Mapper")

# Step 1: Upload CSV
uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Store data in session state
    st.session_state['df'] = df
    
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
    
    # Step 4: Repeat Limit Slider
    repeat_limit = st.slider("Set Repeat Limit", min_value=1, max_value=10, value=min_repeat_limit)
    
    # Step 5: Map Every Row? Checkbox
    map_every_row = st.checkbox("Map Every Row?")

    # Warning if the repeat limit is too low
    row_count = df.shape[0]
    st.warning(f"{row_count} rows detected. You need a repeat link limit of at least {min_repeat_limit} to ensure every URL gets {link_count} links.")
    
    # Step 6: Map Links Button
    if st.button("Map Links"):
        relevance_scores = calculate_relevance_scores(df, title_tag_column)
        df['Relevance Scores'] = relevance_scores.tolist()
        
        new_columns = ['Hub Link URL', 'Hub Link Anchor Text']
        for i in range(1, link_count + 1):
            new_columns.extend([f'Link {i} URL', f'Link {i+1} Anchor Text'])
        
        for col in new_columns:
            df[col] = ""
        
        # Initialize a dictionary to track link usage
        link_usage = {url: 0 for url in df[url_column]}
        
        st.write("Processing Rows...")
        
        # Process each row and calculate top 'link_count' links
        for idx, row in df.iterrows():
            title_scores = row['Relevance Scores']
            sorted_scores_idx = sorted(range(len(title_scores)), key=lambda k: (link_usage[df.at[k, url_column]], -title_scores[k]))

            top_links = []
            # First pass: try to use links within their repeat limit
            for link_idx in sorted_scores_idx[1:]:  # Skip the first one as it's the row itself
                url = df.at[link_idx, url_column]
                if link_usage[url] < repeat limit:
                    top_links.append(link_idx)
                    link_usage[url] += 1
                if len(top_links) == link_count:
                    break
            
            # Second pass: if Map Every Row is checked, ensure every row gets the required number of links, even if it exceeds the repeat limit
            if map_every_row and len(top_links) < link_count:
                for link_idx in sorted_scores_idx[1:]:
                    url = df.at[link_idx, url_column]
                    if len(top_links) == link_count:
                        break
                    if link_idx not in top_links:
                        top_links.append(link_idx)
                        link_usage[url] += 1
            
            if row[hub_spoke_column] == "Spoke":
                df.at[idx, 'Hub Link URL'] = df[df[hub_spoke_column] == "Hub"][url_column].values[0]
                df.at[idx, 'Hub Link Anchor Text'] = df[df[hub_spoke_column] == "Hub"][target_keyword_column].values[0]
            
            for i, link_idx in enumerate(top_links):
                df.at[idx, f'Link {i+1} URL'] = df.at[link_idx, url_column]
                df.at[idx, f'Link {i+1} Anchor Text'] = df.at[link_idx, target_keyword_column]
        
        # Add the link usage count next to the Full URL
        df['Usage Count'] = df['Full URL'].map(link_usage)
        
        # Ensure no row is left without links
        ensure_no_row_without_links(df, link_usage, repeat_limit, link_count)
        
        st.write("Processing Complete!")
        
        # Step 7: Show the processed DataFrame in the Streamlit UI
        st.dataframe(df)
        
        # Enable Generate Visualization and Download Visualization buttons
        st.session_state['link_map_generated'] = True
        
        # Step 8: Download CSV
        output_file_name = uploaded_file.name.replace(".csv", "") + " - Internal Linking Map.csv"
        st.download_button(label="Download CSV", data=df.to_csv(index=False), file_name=output_file_name)

        # Enable Generate Visualization and Download Visualization buttons
        st.session_state['link_map_generated'] = True

# Step 9: Generate Visualization Button (only shown after link map is generated)
if st.session_state.get('link_map_generated', False):
    if st.button("Generate Visualization"):
        st.subheader("Interactive Topic Map")
        
        df = st.session_state.get('df')  # Retrieve the DataFrame from session state
        
        net = create_interactive_graph(df)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:  # Ensure the file has .html extension
            path = tmpfile.name
            net.save_graph(path)
            st.components.v1.html(open(path).read(), height=800, scrolling=True)
            
        # Step 10: Download Visualization Button
        st.download_button(
            label="Download Visualization",
            data=open(path).read(),
            file_name="internal_linking_visualization.html",
            mime="text/html"
        )
