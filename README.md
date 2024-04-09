# CSE-6242-Group-Project
## Gen-AI Anime Recommendation System

### Intro

![image](https://github.com/convoluted-turtle/CSE-6242-Group-Project/assets/33863191/edfb1fa8-1288-4248-a59b-b91f60f5933a)

### Dependencies 

| pypi library | version |
|----------|----------|
| accelerate  | 0.26.1   |
| bitsandbytes  | 0.41.2.post2  |
| faiss-cpu  | 1.8.0  |
| hdbscan   | 0.8.33   |
| IProgress  | 8.1.2  |
| ipywidgets | 0.4 |
| langchain | 0.1.1  |
| langchain-community | 0.0.20 |
| langchain-core  | 0.1.23  |
| langchain-text-splitters | 0.0.1 |
| langcodes  | 3.3.0  |
| langsmith | 0.0.87  |
| llama_cpp_python | 0.2.26  |
| numba | 0.59.0  |
| numpy | 1.26.4 |
| pandas | 2.2.1 |
| peft | 0.7.1 |
| plotly | 5.20.0 |
| sentence-transformers | 2.5.1 |
| spacy | 3.7.4 |
| streamlit | 1.32.2  |
| tokenizers| 0.15.2  |
| transformers | 4.37.2  |
| umap-learn| 0.5.5 |
| google-ai-generativelanguage  | 2.18.0 |
| google-api-core  | 2.18.0 |
| google-auth | 2.28.2 |
| google-generativeai  |  0.4.1 |
| googleapis-common-protos  | 1.63.0 |

### Web Scrapining and Wikipedia API

#### IMDB Actor Scraping

#### Wikipedia Plot API

#### Wikipedia Image API

### Data Joining

#### Joining IMDB

#### Joining Wikipedia

### Collobroative Filtering and Popularity

#### Item-KNN Collabroative Filtering

![image](https://github.com/convoluted-turtle/CSE-6242-Group-Project/assets/33863191/84fe2861-d853-4c19-84b8-228eaebaf56b)

#### Collabroative Filtering Evaluation

#### Thompson Sampling Popular Recommendation

### Vector DataBase Creation 

#### Faiss Vector Database Creation

![image](https://github.com/convoluted-turtle/CSE-6242-Group-Project/assets/33863191/e92ba2d0-673f-4b07-b424-1670a6655ff8)

#### Filter Creation

### Prompting

#### Prompt Template Creation

#### Prompting Evaluation

### Visualization Creation

#### Poster Display

#### Bar Chart

#### UMAP

#### Box and Whiskers

### Application Creation 

#### Streamlit Application

![image](https://github.com/convoluted-turtle/CSE-6242-Group-Project/assets/33863191/e7da3451-d5b8-4ef0-81cc-27bf73d4c503)

#### Docker Image

To rebuild data image containers
```
docker build -f Dockerfile-data -t jonlin8188/cse6242-data:latest .
docker run --rm -it jonlin8188/cse6242-data:latest sh
docker push jonlin8188/cse6242-data:latest
```

To run:
```
docker build -f Dockerfile-app -t jonlin8188/cse6242:latest .
docker run -p 8080:8080 --rm -it --env GOOGLE_API_KEY=ABCDEFG jonlin8188/cse6242:latest 
# Access at http://localhost:8080
docker push jonlin8188/cse6242:latest
```

docker build -f Dockerfile-data -t jonlin8188/cse6242-data:latest .
docker build -t jonlin8188/cse6242:latest .
docker push jonlin8188/cse6242-data:latest
docker push jonlin8188/cse6242:latest













