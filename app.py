import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to calculate relevance score
def calculate_relevance_scores(df, title_tag_column, keyword_column):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df[title_tag_column])
    
    relevance_scores = cosine_similarity(tfidf_matrix, tfidf_matrix)
    df['Relevance Scores'] = relevance_scores.tolist()
    
    # Tie breaker using target keyword relevance if necessary
    if keyword_column in df.columns:
        keyword_vectorizer = TfidfVectorizer()
        keyword_tfidf = keyword_vectorizer.fit_transform(df[keyword_column])
        keyword_scores = cosine_similarity(keyword_tfidf, keyword_tfidf)
        df['Keyword Relevance'] = keyword_scores.tolist()
    
    return df

# Streamlit UI
st.title("Internal Linking Mapper")

# Step 2: Upload CSV
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
    
    # Step 3: Map Links Button
    if st.button("Map Links"):
        df = calculate_relevance_scores(df, title_tag_column, target_keyword_column)
        
        new_columns = ['Hub Link URL', 'Hub Link Anchor Text', 
                       'Link 1 URL', 'Link 1 Anchor Text', 
                       'Link 2 URL', 'Link 2 Anchor Text', 
                       'Link 3 URL', 'Link 3 Anchor Text', 
                       'Link 4 URL', 'Link 4 Anchor Text', 
                       'Link 5 URL', 'Link 5 Anchor Text']
        
        for col in new_columns:
            df[col] = ""
        
        # Step 4: Process each row and calculate top 5 links
        for idx, row in df.iterrows():
            st.write(f"Processing row {idx + 1}...")
            title_scores = row['Relevance Scores']
            sorted_scores_idx = sorted(range(len(title_scores)), key=lambda k: title_scores[k], reverse=True)
            
            top_5 = sorted_scores_idx[1:6]  # Skip the first one as it's the row itself
            
            if row[hub_spoke_column] == "Spoke":
                df.at[idx, 'Hub Link URL'] = df[df[hub_spoke_column] == "Hub"][url_column].values[0]
                df.at[idx, 'Hub Link Anchor Text'] = df[df[hub_spoke_column] == "Hub"][target_keyword_column].values[0]
            
            for i, link_idx in enumerate(top_5):
                df.at[idx, f'Link {i+1} URL'] = df.at[link_idx, url_column]
                df.at[idx, f'Link {i+1} Anchor Text'] = df.at[link_idx, target_keyword_column]
        
        st.write("Processing Complete!")
        
        # Step 5: Show the processed DataFrame in the Streamlit UI
        st.dataframe(df)
        
        # Step 6: Download CSV
        output_file_name = uploaded_file.name.replace(".csv", "") + "- Internal Links.csv"
        st.download_button(label="Download CSV", data=df.to_csv(index=False), file_name=output_file_name)
