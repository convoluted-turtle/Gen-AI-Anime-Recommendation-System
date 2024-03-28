import streamlit as st
from umap import UMAP
import plotly.express as px
from PIL import Image

import hfembeddings as hfembeddings

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

# Load example dataset
df = px.data.iris()
features = df.loc[:, :'petal_width']

umap_2d = UMAP(n_components=2, init='random', random_state=0)
proj_2d = umap_2d.fit_transform(df.loc[:, :'petal_width'])

# Create descending bar plot
fig_bar = px.bar(df, x='petal_width', y='species', color='species',
                 title="Ratings of Popular Anime", orientation='h')
# Sort the bars by petal_width in descending order
fig_bar.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})

# Create vertical box plot
fig_box = px.box(df, x='species', y='petal_width', color='species', 
                 title="Favorites to Producer Distribution", orientation='v')

# Sidebar for chat
with st.sidebar:
    st.markdown("## Chat with AI")
    user_input = st.text_area("Enter your Query here!")
    if st.button("Send"):
        st.write(f"You: {user_input}")
        st.write(f"AI: Here are your recommendations: {user_input}")


# Side title for all three images
st.markdown("<h5 style='text-align: left; margin-top: 30px;'>Anime Recommendations</h5>", unsafe_allow_html=True)
# Main content layout
col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

with col1:
    st.write("<div style='margin-top: 10px;'> </div>", unsafe_allow_html=True)
    st.image(Image.open("2024-03-18_22-38-08.png"), width=300)
    st.write("<p style='position: absolute; bottom:-35px; width: 90%; text-align: center; color: white; background-color: rgba(0, 0, 0, 0.5); padding: 5px;'>Anime 1</p>", unsafe_allow_html=True)



with col2:
    st.write("<div style='margin-top: 10px;'> </div>", unsafe_allow_html=True)
    st.image(Image.open("2024-03-18_22-38-08.png"), width=300)
    st.write("<p style='position: absolute; bottom: -35px; width: 90%; text-align: center; color: white; background-color: rgba(0, 0, 0, 0.5); padding: 5px;'>Anime 2</p>", unsafe_allow_html=True)



with col3:
    st.write("<div style='margin-top: 10px;'> </div>", unsafe_allow_html=True)
    st.image(Image.open("2024-03-18_22-38-08.png"), width=300)
    st.write("<p style='position: absolute; bottom: -35px; width: 90%; text-align: center; color: white; background-color: rgba(0, 0, 0, 0.5); padding: 5px;'>Anime 3</p>", unsafe_allow_html=True)


with col4:
    st.plotly_chart(fig_bar, use_container_width=True)

col5, col6 = st.columns(2)
with col5:
    umap_fig = px.scatter(proj_2d, x=0, y=1, color=df.species, labels={'color': 'Species'})
    umap_fig.update_layout(title="Embedding Space For Recommendations")
    st.plotly_chart(umap_fig, use_container_width=True)


with col6:
    st.plotly_chart(fig_box, use_container_width=True)






