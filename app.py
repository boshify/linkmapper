import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math

# Function to calculate relevance scores
def calculate_relevance_scores(df, title_tag_column):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df[title_tag_column])
    relevance_scores = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return relevance_scores

# Function to calculate minimum link limit needed
def calculate_minimum_link_limit(df, hub_spoke_column, link_count):
    spoke_count = df[df[hub_spoke_column] == "Spoke"].shape[0]
    # Minimum link limit is 'link_count' links per spoke, spread across all other spokes
    return math.ceil(link_count * spoke_count / (spoke_count - 1))

# Function to calculate minimum repeat limit needed
def calculate_minimum_repeat_limit(df, link_count):
    return math.ceil(link_count / 2)  # Example calculation; you can adjust this based on your logic

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
    
    # Calculate the minimum link limit required
    min_link_limit = calculate_minimum_link_limit(df, hub_spoke_column, link_count)
    
    # Calculate the minimum repeat limit required
    min_repeat_limit = calculate_minimum_repeat_limit(df, link_count)
    
    # Step 4: Repeat Limit Slider
    repeat_limit = st.slider("Set Repeat Limit", min_value=1, max_value=10, value=2)
    
    # Step 5: Map Every Row? Checkbox
    map_every_row = st.checkbox("Map Every Row?")

    # Warning if the repeat limit is too low
    row_count = df.shape[0]
    st.warning(f"{row_count} rows detected. You need a link per page limit of at least {min_link_limit} or a repeat link limit of at least {min_repeat_limit} to ensure every URL gets {link_count} links.")
    
    # Step 6: Map Links Button
    if st.button("Map Links"):
        relevance_scores = calculate_relevance_scores(df, title_tag_column)
        df['Relevance Scores'] = relevance_scores.tolist()
        
        new_columns = ['Hub Link URL', 'Hub Link Anchor Text']
        for i in range(1, link_count + 1):
            new_columns.extend([f'Link {i} URL', f'Link {i} Anchor Text'])
        
        for col in new_columns:
            df[col] = ""
        
        # Initialize a dictionary to track link usage
        link_usage = {url: 0 for url in df[url_column]}
        
        st.write("Processing Rows...")
        
        # Process each row and calculate top 'link_count' links
        for idx, row in df.iterrows():
            title_scores = row['Relevance Scores']
            sorted_scores_idx = sorted(range(len(title_scores)), key=lambda k: title_scores[k], reverse=True)
            
            top_links = []
            for link_idx in sorted_scores_idx[1:]:  # Skip the first one as it's the row itself
                url = df.at[link_idx, url_column]
                if link_usage[url] < repeat_limit:
                    top_links.append(link_idx)
                    link_usage[url] += 1
                if len(top_links) == link_count:
                    break
            
            # Ensure every row gets the desired amount of links, even if the limit is reached
            if map_every_row and len(top_links) < link_count:
                for link_idx in sorted_scores_idx[1:]:
                    url = df.at[link_idx, url_column]
                    if len(top_links) == link_count:
                        break
                    if link_idx not in top_links and link_usage[url] < repeat_limit:
                        top_links.append(link_idx)
                        link_usage[url] += 1
            
            if row[hub_spoke_column] == "Spoke":
                df.at[idx, 'Hub Link URL'] = df[df[hub_spoke_column] == "Hub"][url_column].values[0]
                df.at[idx, 'Hub Link Anchor Text'] = df[df[hub_spoke_column] == "Hub"][target_keyword_column].values[0]
            
            for i, link_idx in enumerate(top_links):
                df.at[idx, f'Link {i+1} URL'] = df.at[link_idx, url_column]
                df.at[idx, f'Link {i+1} Anchor Text'] = df.at[link_idx, target_keyword_column]
        
        st.write("Processing Complete!")
        
        # Step 7: Show the processed DataFrame in the Streamlit UI
        st.dataframe(df)
        
        # Step 8: Download CSV
        output_file_name = uploaded_file.name.replace(".csv", "") + "- Internal Linking Map.csv"
        st.download_button(label="Download CSV", data=df.to_csv(index=False), file_name=output_file_name)
