import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Tuple
import pickle 

def create_retriever(faiss_db: str):
    """
    Creates a retriever using a FAISS index.

    Args:
        faiss_db (str): Path to the FAISS index file.

    Returns:
        retriever: A retriever object configured with FAISS index.
    """
    encode_kwargs = {"normalize_embeddings": True}
    
    # Initialize Hugging Face embeddings
    embedding_function = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={"device": "cpu"},
        encode_kwargs=encode_kwargs,
    )

    # Load FAISS index
    db_faiss = FAISS.load_local(faiss_db, embeddings=embedding_function)

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

    # Create retriever object
    retriever = db_faiss.as_retriever(search_kwargs={"k": 50, "filter": filter_tokens})
    return retriever

def load_data(json_file_path: str, cf_pickle_path: str, pop_pickle_path: str) -> Tuple[pd.DataFrame, dict, dict]:
    """
    Load data from JSON and pickle files.

    Args:
        json_file_path (str): Path to the JSON file.
        cf_pickle_path (str): Path to the collaborative filtering pickle file.
        pop_pickle_path (str): Path to the popular recommendations pickle file.

    Returns:
        Tuple[pd.DataFrame, dict, dict]: DataFrame, collaborative filtering recommendations, popular recommendations.
    """
    df = pd.read_json(json_file_path)
    
    with open(cf_pickle_path, 'rb') as f:
        cf_recs = pickle.load(f)

    with open(pop_pickle_path, 'rb') as f:
        pop_recs = pickle.load(f)
    
    return df, cf_recs, pop_recs

def process_recommendations(pop_recs: dict, df: pd.DataFrame, indexes: dict) -> Tuple[List[str], List[str], List[int]]:
    """
    Process recommendations to obtain popular anime descriptions, collaborative filtering recommendations, and VD recommendations.

    Args:
        pop_recs (dict): Popular recommendations.
        df (pd.DataFrame): DataFrame containing anime data.
        indexes (dict): Indexes dictionary.

    Returns:
        Tuple[List[str], List[str], List[int]]: Popular anime descriptions, collaborative filtering recommendations, and VD recommendations.
    """
    def map_anime_ids(anime_id: int) -> List[int]:
        return cf_recs.get(anime_id, [])
    
    df['anime_values'] = df['anime_id'].apply(map_anime_ids)
    
    popular_list = [x['anime_id'] for x in pop_recs]
    popular_anime_descriptions = df[df['anime_id'].isin(popular_list)]['text'].head(5).tolist()
    
    cf_list = list(df[df['anime_id'].isin(list(indexes.keys()))]['anime_values'])
    joined_list = [item for sublist in cf_list if sublist is not None for item in sublist if item is not None] if cf_list else []
    
    vd_recs = list(set(list(indexes.keys())))
    
    return popular_anime_descriptions, joined_list, vd_recs

def get_top3_posters_and_names(df: pd.DataFrame, indexes: dict) -> Tuple[List[str], List[str]]:
    """
    Get top 3 posters and names of anime.

    Args:
        df (pd.DataFrame): DataFrame containing anime data.
        indexes (dict): Indexes dictionary.

    Returns:
        Tuple[List[str], List[str]]: Top 3 posters and names.
    """
    top3_anime_ids = list(indexes.keys())[:3]
    top3_posters = df[df['anime_id'].isin(top3_anime_ids)]['image_y'].tolist()
    top3_names = df[df['anime_id'].isin(top3_anime_ids)]['Name'].tolist()
    return top3_posters, top3_names

def get_recommendations_descriptions(df: pd.DataFrame, joined_list: List[int], pop_recs: List[int], vd_recs: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Get recommendations descriptions.

    Args:
        df (pd.DataFrame): DataFrame containing anime data.
        joined_list (List[int]): Collaborative filtering recommendations.
        pop_recs (List[int]): Popular recommendations.
        vd_recs (List[int]): VD recommendations.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, List[str]]: DataFrame filtered with recommendations, DataFrame filtered with recommendations for joined_list + vd_recs, and list of descriptions.
    """
    recs = df[df['anime_id'].isin(joined_list + pop_recs + vd_recs)]
    recs2 = df[df['anime_id'].isin(joined_list + vd_recs)]
    descriptions = recs['anime_Synopsis'].tolist()
    return recs, recs2, descriptions