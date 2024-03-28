import guidance

from guidance import gen, models
from tqdm import tqdm

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
from utils import TextPreprocessor

textprepo = TextPreprocessor()

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf_embedding_function = HuggingFaceEmbeddings(
    cache_folder='model/all-MiniLM-L6-v2',
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

new_db = FAISS.load_local("embedding/faiss_anime_index_v2", hf_embedding_function, allow_dangerous_deserialization=True)

def are_words_in_string(word_list, input_string):
    for word in word_list:
        if word in input_string:
            return True
    return False

# query = "treasure"
# 

def filter_fn(metadata, query):
    query_token = textprepo.preprocess_text(query)
    return any(word in query for word in list(query_token)) or metadata["score"] > 5.0


# results = new_db.similarity_search(query, filter=filter_fn, k=5)
# results