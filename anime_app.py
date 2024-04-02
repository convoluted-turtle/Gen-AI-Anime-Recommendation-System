import streamlit as st
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from utils.textpreprocessing import TextPreprocessor
from utils.visualizations import streamlit_bar_plot, streamlit_box_whiskers, streamlit_umap
import google.generativeai as genai


import yaml

# Load API key from config.yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

api_key = config["api_key"]

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

userdata = {"GOOGLE_API_KEY": api_key}
GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(model_name='gemini-1.0-pro')
chat = model.start_chat(enable_automatic_function_calling=True)

textprepo = TextPreprocessor()
encode_kwargs = {"normalize_embeddings": True}
embedding_function = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={"device": "cpu"},
    encode_kwargs=encode_kwargs,
)

#Load in data
new_db = FAISS.load_local("/Users/justinvhuang/Desktop/CSE-6242-Group-Project/vector_database_creation/faiss_anime_index_v2", embedding_function)
df = pd.read_json("/Users/justinvhuang/Desktop/CSE-6242-Group-Project/fin_anime_dfv1.json")

# Create sidebar
with st.sidebar:
    st.markdown("## Chat with AI Anime Recommendation")
    initial_query = "I like anime a lot!"
    query = st.text_area("Enter your Query here!", value=initial_query, max_chars = 200)
    query_token = textprepo.preprocess_text(query)
    def filter_tokens(metadata):
        metadata_tokens = metadata.get("tokens", [])
        return any(token in metadata_tokens for token in query_token)
    results = new_db.similarity_search(query, filter= filter_fn, k = 20)
    indexes = {x.metadata['anime_id']: index for index, x in enumerate(results)}
    cf_list = list(df[df['anime_id'].isin(list(indexes.keys()))]['cf_recs'])
    if cf_list is not None:
        joined_list = [item for sublist in cf_list if sublist is not None for item in sublist if item is not None]

    pop_recs = list(df.head(1)['popular_recs'])[0]
    vd_recs = list(indexes.keys())

    recs = df[df['anime_id'].isin(joined_list + pop_recs + vd_recs)]
    recs2 = df[df['anime_id'].isin(joined_list +  vd_recs)]
    descriptions = recs['anime_Synopsis'].tolist()
    response = chat.send_message(f'You are a recommendation AI look at the following animes and summarize it into 5 sentences on why the user might like it: {descriptions}')
    if st.button("Send"):
        st.write(f"You: {query}")
        st.write(f"AI: Here are your recommendations: {response.text}")


top_anime_rating = recs2[recs2['anime_Score']!='UNKNOWN'].sort_values(by='anime_Score', ascending=False).head(5)
top_studios = recs2.sort_values(by='Favorites', ascending=False).head(5)
top_anime_rating['anime_Score'] = top_anime_rating['anime_Score'].astype(float)

recs_umap = recs[['Studios', 'anime_Synopsis', 'Name', 'anime_id']]
recs_umap['rec_label'] = ''

# Iterate through the lists and update the 'rec_label' column
for rec_type, lst in [('collab_filter', joined_list), ('vector_rec', vd_recs), ('pop_rec', pop_recs)]:
    recs_umap.loc[recs_umap['anime_id'].isin(lst), 'rec_label'] = rec_type
new_row = {'Studios': 'user query', 'anime_Synopsis': query, 'Name': 'user query', 'anime_id': 'none', 'rec_label': 'none'}
recs_umap = pd.concat([pd.DataFrame(new_row, index=[0]), recs_umap], ignore_index=True)

fig_bar = streamlit_bar_plot(top_anime_rating)

fig_box = streamlit_box_whiskers(df[df['Studios'].isin(set(list(top_studios['Studios'])))])

fig_umap, closet_anime_ids = streamlit_umap(recs_umap)


# Side title for all three images
st.markdown("<h5 style='text-align: left; margin-top: 30px;'>Anime Recommendations</h5>", unsafe_allow_html=True)
# Main content layout
col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

with col1:
    st.write("<div style='margin-top: 10px;'> </div>", unsafe_allow_html=True)
    image_url = "https://upload.wikimedia.org/wikipedia/en/8/85/Muramasa_The_Demon_Blade.jpg"
    st.image(image_url, width=300)
    st.markdown(
        f"<p style='width: 300px; text-align: center; color: white; background-color: rgba(0, 0, 0, 0.5); padding: 5px;'>{closet_anime_ids[0]}</p>",
        unsafe_allow_html=True
    )

with col2:
    st.write("<div style='margin-top: 10px;'> </div>", unsafe_allow_html=True)
    image_url = "https://upload.wikimedia.org/wikipedia/en/7/71/Kyoshiro_to_Towa_no_Sora_volume_1_cover.jpg"
    st.image(image_url, width=300)
    st.markdown(
        f"<p style='width: 300px; text-align: center; color: white; background-color: rgba(0, 0, 0, 0.5); padding: 5px;'>{closet_anime_ids[1]}</p>",
        unsafe_allow_html=True
    )


with col3:
    st.write("<div style='margin-top: 10px;'> </div>", unsafe_allow_html=True)
    image_url = "https://static.wikia.nocookie.net/initiald/images/e/ec/First_Stage_logo.png"
    st.image(image_url, width=300)
    st.markdown(
        f"<p style='width: 300px; text-align: center; color: white; background-color: rgba(0, 0, 0, 0.5); padding: 5px;'>{closet_anime_ids[2]}</p>",
        unsafe_allow_html=True
    )


with col4:
    st.plotly_chart(fig_bar, use_container_width=True)

col5, col6 = st.columns(2)
with col5:
    st.plotly_chart(fig_umap, use_container_width=True)


with col6:
    st.plotly_chart(fig_box, use_container_width=True)






