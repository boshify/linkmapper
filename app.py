import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math

# Function to calculate relevance scores
def calculate_relevance_scores(df, column1, column2):
    vectorizer = TfidfVectorizer()
    combined_columns = df[column1] + " " + df[column2]
    tfidf_matrix = vectorizer.fit_transform(combined_columns)
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
                if url != row['Full URL'] and (link_usage[url] < repeat_limit or links_added < link_count):
                    for i in range(link_count):
                        if pd.isnull(row[f'Link {i+1} URL']):
                            df.at[idx, f'Link {i+1} URL'] = url
                            df.at[idx, f'Link {i+1} Anchor Text'] = df.loc[df['Full URL'] == url, 'Target Keyword'].values[0]
                            link_usage[url] += 1
                            links_added += 1
                            break
                if links_added == link_count:
                    break

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
    
    # Step 4: Repeat Limit Slider
    repeat_limit = st.slider("Set Repeat Limit", min_value=1, max_value=10, value=min_repeat_limit)
    
    # Step 5: Map Every Row? Checkbox
    map_every_row = st.checkbox("Map Every Row?")

    # Warning if the repeat limit is too low
    row_count = df.shape[0]
    st.warning(f"{row_count} rows detected. You need a repeat link limit of at least {min_repeat_limit} to ensure every URL gets {link_count} links.")
    
    # Step 6: Calculate Relevance Score By
    st.subheader("Calculate Relevance Score By")
    st.write("Select two columns to calculate the relevance score.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        relevance_column1 = st.selectbox("First Column", df.columns)
    
    with col2:
        relevance_column2 = st.selectbox("Second Column", df.columns)
    
    # Step 7: Map Links Button
    if st.button("Map Links"):
        relevance_scores = calculate_relevance_scores(df, relevance_column1, relevance_column2)
        df['Relevance Scores'] = relevance_scores.tolist()
        
        # Define the correct columns for URLs and Anchor Texts
        for i in range(1, link_count + 1):
            df[f'Link {i} URL'] = ""
            df[f'Link {i+1} Anchor Text'] = ""
        
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
                if url != row['Full URL'] and link_usage[url] < repeat_limit:
                    top_links.append(link_idx)
                    link_usage[url] += 1
                if len(top_links) == link_count:
                    break
            
            # Second pass: if Map Every Row is checked, ensure every row gets the required number of links, even if it exceeds the repeat limit
            if map_every_row and len(top_links) < link_count:
                for link_idx in sorted_scores_idx[1:]:
                    url = df.at[link_idx, url_column]
                    if url != row['Full URL'] and len(top_links) < link_count:
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
        
        # Step 8: Show the processed DataFrame in the Streamlit UI
        st.dataframe(df)
        
        # Step 9: Download CSV
        output_file_name = uploaded_file.name.replace(".csv", "") + " - Internal Linking Map.csv"
        st.download_button(label="Download CSV", data=df.to_csv(index=False), file_name=output_file_name)
