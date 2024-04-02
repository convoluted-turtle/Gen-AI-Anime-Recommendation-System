import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, List

def streamlit_bar_plot(df: pd.DataFrame) -> go.Figure:
    """
    Generate a bar plot for anime ratings.

    Args:
    - df (pd.DataFrame): DataFrame containing anime ratings data.

    Returns:
    - fig (go.Figure): Plotly bar plot.
    """
    # Reset bar plot to default state
    fig = px.bar(df.sort_values(by="anime_Score", ascending=True), x='anime_Score', y='Name', title="Ratings of Popular Anime", orientation='h')
    
    # Determine buffer region size
    buffer_region = 0.1 * (df["anime_Score"].max() - df["anime_Score"].min())
    
    # Set the range of the x-axis with a buffer region below the minimum score and above the maximum score
    fig.update_xaxes(range=[df["anime_Score"].min() - buffer_region, df["anime_Score"].max() + buffer_region])

    return fig

def streamlit_box_whiskers(df: pd.DataFrame) -> go.Figure:
    """
    Generate a box plot for anime favorites distribution by studios.

    Args:
    - df (pd.DataFrame): DataFrame containing anime favorites and studios data.

    Returns:
    - fig_box (go.Figure): Plotly box plot.
    """
    # Create vertical box plot for the filtered data
    fig_box = px.box(df, x='Studios', y='Favorites', color='Studios', 
                     title="Favorites to Studios Distribution", orientation='v', custom_data=['Name'])

    # Add text labels for maximum values
    fig_box.update_traces(
        hovertemplate="<b>Name:</b> %{customdata[0]}<br><b>Favorites:</b> %{y}",
        selector=dict(type='box')
    )
    
    return fig_box

def streamlit_umap(recs_umap: pd.DataFrame) -> Tuple[go.Figure, List[str]]:
    """
    Generate a UMAP plot for anime recommendations.

    Args:
    - recs_umap (pd.DataFrame): DataFrame containing UMAP data for anime recommendations.

    Returns:
    - fig_umap (go.Figure): Plotly scatter plot for UMAP.
    - closest_anime_ids (List[str]): List of anime IDs for the closest points to the marked point.
    """
    # Load a pre-trained Sentence Transformer model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Encode the anime synopsis using the Sentence Transformer model
    embeddings = model.encode(recs_umap['anime_Synopsis'].tolist())

    # Apply UMAP for dimensionality reduction
    umap_model = UMAP(n_components=2, n_neighbors=5, min_dist=0.05)
    umap_result = umap_model.fit_transform(embeddings)

    # Convert UMAP result to DataFrame
    umap_df = pd.DataFrame(umap_result, columns=['UMAP_1', 'UMAP_2'])

    # Add 'Studios' and 'Name' columns to the UMAP DataFrame
    umap_df['Studios'] = recs_umap['Studios'].tolist()
    umap_df['Name'] = recs_umap['Name'].tolist()
    umap_df['rec_label'] = recs_umap['rec_label'].tolist()
    umap_df['anime_id'] = recs_umap['anime_id'].tolist()

    # Plot the UMAP with color by 'rec_label'
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

    # Collect anime Name of the three closest points
    closest_anime_names = umap_df.loc[indices[0][1:], 'Name'].tolist()

     # Collect anime IDs of the three closest points
    closest_anime_ids = umap_df.loc[indices[0][1:], 'anime_id'].tolist()

    # Plot red X symbol on the closest points (excluding the marked point)
    for i, idx in enumerate(indices[0][1:], start=1):
        target_x = umap_df.loc[idx, 'UMAP_1']
        target_y = umap_df.loc[idx, 'UMAP_2']
        fig_umap.add_trace(go.Scatter(x=[target_x], y=[target_y], mode='markers', showlegend=True,
                                      marker=dict(symbol='x', size=10, color='red'), name=f'rec {i}'))

    # Update hover template to include only 'Name' and 'Studios'
    fig_umap.update_traces(customdata=umap_df[['Studios', 'Name']],
                            hovertemplate="<b>%{customdata[1]}</b><br>" +
                                          "Studios: %{customdata[0]}<br>" +
                                          "<extra></extra>")

    return fig_umap, closest_anime_names, closest_anime_ids
