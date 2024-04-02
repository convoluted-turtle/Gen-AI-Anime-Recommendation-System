import streamlit as st
import pandas as pd
import plotly.express as px
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from utils.textpreprocessing import TextPreprocessor
from umap import UMAP
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
from PIL import Image
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
    def filter_fn(metadata):
        return any(word in query for word in list(query_token))
    results = new_db.similarity_search(query, filter= filter_fn, k = 5)
    intros = set([x.page_content for x in results])
    response = chat.send_message(f'You are a recommendation AI look at the following animes and summarize it into 3 sentences on why the user might like it: {intros}')
    if st.button("Send"):
        st.write(f"You: {query}")
        st.write(f"AI: Here are your recommendations: {response.text}")


indexes = {x.metadata['anime_id']: index for index, x in enumerate(results)}
cf_list = list(df[df['anime_id'].isin(list(indexes.keys()))]['cf_recs'])
if cf_list is not None:
    joined_list = [item for sublist in cf_list if sublist is not None for item in sublist if item is not None]

pop_recs = list(df.head(1)['popular_recs'])[0]
vd_recs = list(indexes.keys())

recs = df[df['anime_id'].isin(joined_list + pop_recs + vd_recs)]
recs2 = df[df['anime_id'].isin(joined_list +  vd_recs)]

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


# Load a pre-trained Sentence Transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Encode the anime synopsis using the Sentence Transformer model
embeddings = model.encode(recs_umap['anime_Synopsis'].tolist())

# Apply UMAP for dimensionality reduction
umap_model = UMAP(n_components=2, n_neighbors=5, min_dist=0.05)
umap_result = umap_model.fit_transform(embeddings)

# Convert UMAP result to DataFrame
umap_df = pd.DataFrame(umap_result, columns=['UMAP_1', 'UMAP_2'])

# Add 'Studios' and 'anime_id' columns to the UMAP DataFrame
umap_df['Studios'] = recs_umap['Studios'].tolist()
umap_df['Name'] = recs_umap['Name'].tolist()
umap_df['rec_label'] = recs_umap['rec_label'].tolist()
umap_df['anime_id'] = recs_umap['anime_id'].tolist()

# Plot the UMAP with color by 'rec_label' and hover information including 'Studios' and 'Name'
fig_umap = px.scatter(umap_df, x='UMAP_1', y='UMAP_2', color='rec_label', 
                      hover_data={'Studios': True, 'Name': True},
                      title='UMAP of Anime Recommendations from Collab Filter, Vector Database and Popular Recommendations')

# Modify the marker symbol for points labeled 'pop_rec' to be a star with yellow color and bigger size
fig_umap.for_each_trace(lambda t: t.update(marker=dict(symbol='star', size=12, color='yellow') if t.name == 'pop_rec' else {}))

# Add annotation for a specific point
x_coord = umap_df.loc[0, 'UMAP_1']
y_coord = umap_df.loc[0, 'UMAP_2']
fig_umap.add_annotation(x=x_coord, y=y_coord, text="X", showarrow=True, font=dict(color="purple", size=20))

# Find the three closest points to the marked point
nn_model = NearestNeighbors(n_neighbors=4, metric='euclidean')
nn_model.fit(umap_result)
distances, indices = nn_model.kneighbors([umap_result[0]])

# Collect anime IDs of the three closest points
closest_anime_ids = umap_df.loc[indices[0][1:], 'Name'].tolist()

# Plot red X symbol on the closest points (excluding the marked point)
for i, idx in enumerate(indices[0][1:], start=1):
    target_x = umap_df.loc[idx, 'UMAP_1']
    target_y = umap_df.loc[idx, 'UMAP_2']
    fig_umap.add_trace(go.Scatter(x=[target_x], y=[target_y], mode='markers', showlegend=True,
                                  marker=dict(symbol='x', size=10, color='red'), name=f'rec {i}'))

# Create descending bar plot
fig_bar = px.bar(top_anime_rating, x='anime_Score', y='Name', color='Name',
                 title="Ratings of Popular Anime", orientation='h')

# Determine the buffer region
buffer_region = 0.1 * (top_anime_rating['anime_Score'].max() - top_anime_rating['anime_Score'].min())

# Set the range of the x-axis with a buffer region below the minimum score and above the maximum score
fig_bar.update_xaxes(range=[top_anime_rating['anime_Score'].min() - buffer_region, top_anime_rating['anime_Score'].max() + buffer_region])
# Define shades of blue
blue_palette = ['#aec7e8', '#7b9fcf', '#1f77b4', '#03539e', '#003f5c']
# Set colors for bars
fig_bar.update_traces(marker_color=blue_palette)

# Create vertical box plot for the filtered data
fig_box = px.box(df[df['Studios'].isin(set(list(top_studios['Studios'])))], 
                 x='Studios', 
                 y='Favorites', 
                 color='Studios', 
                 title="Favorites to Studios Distribution", 
                 orientation='v',
                 custom_data=['Name'])

# Get the indices of maximum values
max_indices = df.groupby('Studios')['Favorites'].idxmax()

# Add text labels for maximum values
fig_box.update_traces(
    hovertemplate="<b>Name:</b> %{customdata[0]}<br><b>Favorites:</b> %{y}",
    selector=dict(type='box')
)


# Side title for all three images
st.markdown("<h5 style='text-align: left; margin-top: 30px;'>Anime Recommendations</h5>", unsafe_allow_html=True)
# Main content layout
col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

with col1:
    st.write("<div style='margin-top: 10px;'> </div>", unsafe_allow_html=True)
    image_url = "https://upload.wikimedia.org/wikipedia/en/8/85/Muramasa_The_Demon_Blade.jpg"
    st.image(image_url, width=300)
    st.markdown(
        f"<p style='width: 300px; text-align: center; color: white; background-color: rgba(0, 0, 0, 0.5); padding: 5px;'>{closest_anime_ids[0]}</p>",
        unsafe_allow_html=True
    )



with col2:
    st.write("<div style='margin-top: 10px;'> </div>", unsafe_allow_html=True)
    image_url = "https://upload.wikimedia.org/wikipedia/en/7/71/Kyoshiro_to_Towa_no_Sora_volume_1_cover.jpg"
    st.image(image_url, width=300)
    st.markdown(
        f"<p style='width: 300px; text-align: center; color: white; background-color: rgba(0, 0, 0, 0.5); padding: 5px;'>{closest_anime_ids[1]}</p>",
        unsafe_allow_html=True
    )
    # st.write("<div style='margin-top: 10px;'> </div>", unsafe_allow_html=True)
    # st.image(Image.open("https://upload.wikimedia.org/wikipedia/en/7/71/Kyoshiro_to_Towa_no_Sora_volume_1_cover.jpg"), width=300)
    # st.write("<p style='position: absolute; bottom: -35px; width: 90%; text-align: center; color: white; background-color: rgba(0, 0, 0, 0.5); padding: 5px;'>Anime 2</p>", unsafe_allow_html=True)



with col3:
    st.write("<div style='margin-top: 10px;'> </div>", unsafe_allow_html=True)
    image_url = "https://static.wikia.nocookie.net/initiald/images/e/ec/First_Stage_logo.png"
    st.image(image_url, width=300)
    st.markdown(
        f"<p style='width: 300px; text-align: center; color: white; background-color: rgba(0, 0, 0, 0.5); padding: 5px;'>{closest_anime_ids[2]}</p>",
        unsafe_allow_html=True
    )


with col4:
    st.plotly_chart(fig_bar, use_container_width=True)

col5, col6 = st.columns(2)
with col5:
    st.plotly_chart(fig_umap, use_container_width=True)


with col6:
    st.plotly_chart(fig_box, use_container_width=True)






