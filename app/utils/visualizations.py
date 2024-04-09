import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, List
import numpy as np
from scipy.spatial.distance import cdist

def streamlit_bar_plot(df: pd.DataFrame) -> go.Figure:
    """
    Generate a bar plot for anime ratings.

    Args:
    - df (pd.DataFrame): DataFrame containing anime ratings data.

    Returns:
    - fig (go.Figure): Plotly bar plot.
    """
    # Reset bar plot to default state
    fig = px.bar(df.sort_values(by="anime_Score", ascending=True), x='anime_Score', y='Name', title="Ratings of Popular Anime from Collabroative Filtering and Dense Vector Cosine Similarity", orientation='h')
    
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
                     title="Favorites to Studios Distribution from Collabroative Filtering and Dense Vector Cosine Similarity", orientation='v', custom_data=['Name'])

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
    embeddings_synopsis = model.encode(recs_umap['anime_Synopsis'].tolist())
    embedding_producers = model.encode(recs_umap['Producers'].tolist())
    embedding_aired = model.encode(recs_umap['Aired'].tolist())
    embedding_studios = model.encode(recs_umap['Studios'].tolist())
    embedding_actors = model.encode(recs_umap['imdb_name_basics_primaryName'].tolist())
    embedding_genres = model.encode(recs_umap['Genres'].tolist())
    ratings_encoded = np.array(recs_umap['anime_Score']).reshape(-1, 1)
    combined_features = np.concatenate([embeddings_synopsis,embedding_producers,embedding_aired,embedding_studios,embedding_actors,ratings_encoded,embedding_genres], axis = 1)

   # Apply UMAP for dimensionality reduction
    umap_model = UMAP(n_components=2, n_neighbors=5, min_dist=0.05,  metric= 'euclidean',random_state=0)
    umap_result = umap_model.fit_transform(combined_features)

    # Convert UMAP result to DataFrame
    umap_df = pd.DataFrame(umap_result, columns=['UMAP_1', 'UMAP_2'])

    # Add 'Studios' and 'Name' columns to the UMAP DataFrame
    umap_df['Studios'] = recs_umap['Studios'].tolist()
    umap_df['Name'] = recs_umap['Name'].tolist()
    umap_df['rec_label'] = recs_umap['rec_label'].tolist()
    umap_df['anime_id'] = recs_umap['anime_id'].tolist()
    umap_df['actors'] = recs_umap['imdb_name_basics_primaryName'].tolist()
    umap_df['producers'] = recs_umap['Producers'].tolist()

    # Plot the UMAP with color by 'rec_label'
    fig_umap = px.scatter(umap_df, x='UMAP_1', y='UMAP_2', color='rec_label', 
                            hover_data={'Studios': True, 'Name': True, 'actors': True, 'producers': True},
                            title='UMAP of Anime Recommendations from Collab Filter, Vector Database and Popular Recommendations')

    # Modify the marker symbol for points labeled 'pop_rec' to be a star with yellow color and bigger size
    fig_umap.for_each_trace(lambda t: t.update(marker=dict(symbol='star', size=12, color='blue')) if t.name == 'pop_rec' else None)

    # Add annotation for a specific point
    x_coord = umap_df.loc[0, 'UMAP_1']
    y_coord = umap_df.loc[0, 'UMAP_2']
    fig_umap.add_annotation(x=x_coord, y=y_coord, text="X", showarrow=True, font=dict(color="purple", size=20))
    # Calculate pairwise distances between row 0 and all other rows
    distances = cdist(umap_df[['UMAP_1', 'UMAP_2']].iloc[[0]], umap_df[['UMAP_1', 'UMAP_2']], metric='euclidean')[0]
    # Sort distances and get the indices of the three closest rows (excluding row 0 itself)
    closest_indices = np.argsort(distances)[1:4]

    # Extract the closest rows based on the indices
    closest_rows = umap_df.iloc[closest_indices]
    closest_anime_names = closest_rows['Name'].tolist()
    closest_anime_ids = closest_rows['anime_id'].tolist()

    for i, (index, row) in enumerate(closest_rows.iterrows(), start=1):
        x_coord = row['UMAP_1']
        y_coord = row['UMAP_2']
        fig_umap.add_trace(go.Scatter(x=[x_coord], y=[y_coord], mode='markers', marker=dict(symbol='x', size=10, color='red'), name=f'rec {i}'))


    # Remove x-axis and y-axis labels
    fig_umap.update_layout(xaxis=dict(title_text=''), yaxis=dict(title_text=''))
    # Remove x-axis and y-axis labels, ticks, and gridlines
    fig_umap.update_layout(xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False))
    # Set light gray background with higher opacity
    fig_umap.update_layout(plot_bgcolor='rgba(220, 220, 220, 0.1)')

    return fig_umap, closest_anime_names, closest_anime_ids
