import streamlit as st
import pandas as pd
from utils.textpreprocessing import TextPreprocessor
from utils.data_manipulation import (create_retriever, 
                                         load_data, 
                                         process_recommendations, 
                                         get_top3_posters_and_names, 
                                         get_recommendations_descriptions)
textprepo = TextPreprocessor()
from utils.prompt import (load_llm,
                          format_docs,
                          get_template,
                          popular_recs)
from utils.visualizations import streamlit_bar_plot, streamlit_box_whiskers, streamlit_umap
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from PIL import Image


sbert = "sentence-transformers/all-MiniLM-L6-v2"
vdb = "/Users/justinvhuang/Desktop/CSE-6242-Group-Project/app/faiss_anime_index_v3"
json_file_path = "/Users/justinvhuang/Desktop/CSE-6242-Group-Project/app/fin_anime_dfv3.json"
cf_pickle_path = "/Users/justinvhuang/Desktop/CSE-6242-Group-Project/app/anime_recommendations_item_knn_CF_10k_num_fin.pkl"
pop_pickle_path = "/Users/justinvhuang/Desktop/CSE-6242-Group-Project/app/popular_dict_10.pkl"
llm_model = "/Users/justinvhuang/Desktop/CSE-6242-Group-Project/app/config.yaml"

db_faiss = create_retriever(vdb, sbert)
df, cf_recs, pop_recs = load_data(json_file_path, cf_pickle_path, pop_pickle_path)
llm = load_llm(llm_model)
custom_rag_prompt = get_template()

def filter_tokens(metadata: dict) -> bool:
    """
    Filter function to apply on retrieved documents based on metadata.

    Args:
        metadata (dict): Metadata of the document.
        query_token (list): List of tokens to filter.

    Returns:
        bool: True if the document passes the filter, False otherwise.
    """
    metadata_tokens = metadata.get("tokens", [])
    metadata_studio = metadata.get("studio", [])
    metadata_producer = metadata.get("producer", [])
    metadata_licensors = metadata.get("licensors", [])
    metadata_genre = metadata.get("genre", [])

    return (
        any(token in metadata_tokens for token in query_token)
        or metadata.get("score", 0.0) > 5.0
        or any(token in metadata_studio for token in query_token)
        or any(token in metadata_producer for token in query_token)
        or any(token in metadata_licensors for token in query_token)
        or any(token in metadata_genre for token in query_token)
    )

retriever = db_faiss.as_retriever(search_kwargs={"k": 50, "filter": filter_tokens})


# Set page title and layout
st.set_page_config(page_title='CSE 6242: Casual Correlations- GenAI Anime Recommendations', layout='wide')

# Custom CSS to inject
st.markdown("""
<style>
    .reportview-container {
        background: white;
    }
    .sidebar .sidebar-content {
        background-color: teal;
    }
    header {
        background-color: goldenrod;
        color: black;
        padding: 10px;
        line-height: 50px;
        font-size: 25px;
        text-align: center;
    }
    .Widget>label {
        color: white;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

# Title in the golden banner
st.markdown('<header>CSE 6242: Casual Correlations- GenAI Anime Recommendations</header>', unsafe_allow_html=True)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)


query = 'what are some good space pirate anime'
query_token = textprepo.preprocess_text(query)
print(rag_chain.invoke(query))

results = retriever.get_relevant_documents(query)

with st.sidebar:
    st.markdown("## Chat with AI Anime Recommendation")
    studios = st.text_input("Studios", "")
    animescore = st.text_input("Anime Score", "8.0")
    producer = st.text_input("Producer", "")
    actors = st.text_input('Actors', "")
    genre = st.text_input('Genre', "")
    if not isinstance(studios, str):
        st.warning("Please enter a valid string for Studios.")
    if not isinstance(producer, str):
        st.warning("Please enter a valid string for Producer.")
    if not isinstance(actors, str):
        st.warning("Please enter a valid string for Actors.")
    if not isinstance(genre, str):
        st.warning("Please enter a valid string for Genre.")
    
    try:
        initial_query = "I like anime a lot!"
        query = st.text_area("Enter your Query here!", value=initial_query, max_chars=200)
        query_token = textprepo.preprocess_text(query)
        results = retriever.get_relevant_documents(query)
        indexes = {x.metadata['anime_id']: index for index, x in enumerate(results)}
        pop_recs, popular_anime_descriptions, joined_list, vd_recs = process_recommendations(pop_recs, df, indexes, cf_recs)
        top3_posters, closet_anime_name = get_top3_posters_and_names(df, indexes)
        recs, recs2, descriptions = get_recommendations_descriptions(df, joined_list, pop_recs, vd_recs)
        response = rag_chain.invoke(query)
    except Exception as e:
        response = popular_recs()
        cf_list = list(df[df['anime_id'].isin(pop_recs)]['anime_values'])
        joined_list = [item for sublist in cf_list for item in sublist]
        recs = df[df['anime_id'].isin(pop_recs+joined_list)]
        recs2 = df[df['anime_id'].isin(pop_recs+joined_list)]
    
   
    if st.button("Send"):
        st.write(f"You: {query}")
        st.sidebar.write(f"AI: Here are your recommendations: \n \n {response}")


top_anime_rating = recs2[recs2['anime_Score']!='UNKNOWN'].sort_values(by='anime_Score', ascending=False).head(10)
top_studios = recs2.sort_values(by='Favorites', ascending=False).head(5)
top_anime_rating['anime_Score'] = top_anime_rating['anime_Score'].astype(float)

recs_umap = recs[['Studios', 'anime_Synopsis', 'Name', 'anime_id','Image URL', 'Producers', 'anime_Score', 'Source', 'Favorites', 'Members', 'Aired', 'imdb_name_basics_primaryName', 'Genres']]
# Initialize 'rec_label' column with empty strings
recs_umap['rec_label'] = ''

# Iterate through the lists and update the 'rec_label' column
for rec_type, lst in [('collab_filter', joined_list), ('vector_rec', vd_recs), ('pop_rec', pop_recs)]:
    recs_umap.loc[recs_umap['anime_id'].isin(lst), 'rec_label'] = rec_type
new_row = {'Studios': studios, 'anime_Synopsis': query, 'Name': '', 'anime_id': '', 'rec_label': 'none', 'Image URL': 'none', 'Producers': producer, 'anime_Score': animescore, 'Source': '', 'Favorites':'', 'Members':'', ' Aired':'', 'imdb_name_basics_primaryName': actors, 'Genres': genre}
recs_umap = pd.concat([pd.DataFrame(new_row, index=[0]), recs_umap], ignore_index=True)

fig_bar = streamlit_bar_plot(top_anime_rating)

fig_box = streamlit_box_whiskers(df[df['Studios'].isin(set(list(top_studios['Studios'])))])

fig_umap, closet_anime_name, closet_anime_ids = streamlit_umap(recs_umap)


# Side title for all three images
st.markdown("<h5 style='text-align: left; margin-top: 30px;'>Anime Recommendations according UMAP</h5>", unsafe_allow_html=True)
# Main content layout
col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

with col1:
    st.write("<div style='margin-top: 10px;'> </div>", unsafe_allow_html=True)
    #image_url = top3_posters[0]
    image_url = df[df['anime_id'] == closet_anime_ids[0]]['Image URL'].tolist()[0]
    st.markdown(f"<img src='{image_url}' width='300' height='400'>", unsafe_allow_html=True)

    st.markdown(
        f"<p style='width: 300px; text-align: center; color: white; background-color: rgba(0, 0, 0, 0.5); padding: 5px;'>{closet_anime_name[0]}</p>",
        unsafe_allow_html=True
    )

with col2:
    st.write("<div style='margin-top: 10px;'> </div>", unsafe_allow_html=True)
    #image_url = top3_posters[1]
    image_url = df[df['anime_id'] == closet_anime_ids[1]]['Image URL'].tolist()[0]
    st.markdown(f"<img src='{image_url}' width='300' height='400'>", unsafe_allow_html=True)

    st.markdown(
        f"<p style='width: 300px; text-align: center; color: white; background-color: rgba(0, 0, 0, 0.5); padding: 5px;'>{closet_anime_name[1]}</p>",
        unsafe_allow_html=True
    )

with col3:
    st.write("<div style='margin-top: 10px;'> </div>", unsafe_allow_html=True)
    #image_url = top3_posters[2]
    image_url = df[df['anime_id'] == closet_anime_ids[2]]['Image URL'].tolist()[0]
    st.markdown(f"<img src='{image_url}' width='300' height='400'>", unsafe_allow_html=True)

    st.markdown(
        f"<p style='width: 300px; text-align: center; color: white; background-color: rgba(0, 0, 0, 0.5); padding: 5px;'>{closet_anime_name[2]}</p>",
        unsafe_allow_html=True
    )

with col4:
    st.plotly_chart(fig_bar, use_container_width=True)

col5, col6 = st.columns(2)
with col5:
    st.plotly_chart(fig_umap, use_container_width=True)


with col6:
    st.plotly_chart(fig_box, use_container_width=True)






