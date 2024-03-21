import streamlit as st
from umap import UMAP
import plotly.express as px
from PIL import Image

# Set the page to wide mode
st.set_page_config(layout="wide")

# Load example dataset
df = px.data.iris()
features = df.loc[:, :'petal_width']

# UMAP for 2D
umap_2d = UMAP(n_components=2, init='random', random_state=0)
proj_2d = umap_2d.fit_transform(features)

# Create plots
fig_bar = px.bar(df, y='species', x='petal_width', color='species', orientation='h')
fig_2d = px.scatter(proj_2d, x=0, y=1, color=df.species, labels={'color': 'Species'})
fig_box = px.box(df, y='species', x='petal_width', color='species', orientation='h')

# Sidebar for chat
with st.sidebar:
    with st.expander("Chat with AI"):
        user_input = st.text_area("Type your message here")
        if st.button("Send"):
            st.write(f"You: {user_input}")
            # In a real app, here you would send the input to a model and get a response
            st.write(f"AI: Echoing what you said: {user_input}")

# Use the full page instead of a narrow central column
col1, col2, col3, col4 = st.columns([1,1,1,2])

with col1:
    st.subheader("Anime Image 1")
    st.image(Image.open("/Users/justinvhuang/Desktop/CSE-6242-Group-Project/de6bcbb7-1953-48e4-9de7-f9a3a31193ee.webp"), width=400)

with col2:
    st.subheader("Anime Image 2")
    st.image(Image.open("/Users/justinvhuang/Desktop/CSE-6242-Group-Project/de6bcbb7-1953-48e4-9de7-f9a3a31193ee.webp"), width=400)

with col3:
    st.subheader("Anime Image 3")
    st.image(Image.open("/Users/justinvhuang/Desktop/CSE-6242-Group-Project/de6bcbb7-1953-48e4-9de7-f9a3a31193ee.webp"), width=400)

with col4:
    st.subheader("Horizontal Bar Chart")
    st.plotly_chart(fig_bar, use_container_width=True)

# UMAP and Box plots below images
col5, col6 = st.columns([2, 2])
with col5:
    st.subheader("2D UMAP Projection")
    st.plotly_chart(fig_2d, use_container_width=True)

with col6:
    st.subheader("Box and Whiskers Plot")
    st.plotly_chart(fig_box, use_container_width=True)






