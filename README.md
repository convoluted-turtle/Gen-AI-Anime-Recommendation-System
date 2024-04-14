# CSE-6242-Group-Project
## Gen-AI Anime Recommendation System

### Intro

<img src="https://github.com/convoluted-turtle/CSE-6242-Group-Project/assets/33863191/edfb1fa8-1288-4248-a59b-b91f60f5933a" alt="intro" width="700">

---

### Dependencies 

<div align="center">

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

</div>

---

### Web Scrapining and Wikipedia API

#### IMDB Actor Scraping

#### Wikipedia Plot API

#### Wikipedia Image API

---

### Data Joining

#### Joining IMDB

#### Joining Wikipedia

---

### Collobroative Filtering and Popularity

#### Item-KNN Collabroative Filtering

<img src="https://github.com/convoluted-turtle/CSE-6242-Group-Project/assets/33863191/84fe2861-d853-4c19-84b8-228eaebaf56b" alt="item-knn" width="450">


#### Collabroative Filtering Evaluation

<div align="center">

| Model                | RMSE  | F1-score | MAP   | NDCG@5 |
|----------------------|-------|----------|-------|--------|
| ItemKNN-Cosine       | 1.2106| 0.0153   | 0.0315| 0.0590 |
| ItemKNN-Pearson      | 1.2147| 0.0153   | 0.0311| 0.0545 |
| ItemKNN-Adjusted Cosine | 1.2135| 0.0153 | 0.0314| 0.0657 |

</div>

#### Thompson Sampling Popular Recommendation

---

### Vector DataBase Creation 

#### Faiss Vector Database Creation

![image](https://github.com/convoluted-turtle/CSE-6242-Group-Project/assets/33863191/e92ba2d0-673f-4b07-b424-1670a6655ff8)

#### Filter Creation

---

### Prompting

#### Prompt Template Creation

<img src="https://github.com/convoluted-turtle/CSE-6242-Group-Project/assets/33863191/a046c012-2bb7-46d5-951f-62e5c1dbba65" alt="prompting" width="600">


#### Prompting Evaluation

* **Rouge**

| Prompt/Metric                | Rouge1  | Rouge2 | RougeL   | RougeLsum |
|----------------------|-------|----------|-------|--------|
| Zero-Shot       | 0.22676| 0.07376  | 0.15299| 0.18761 |
| One-Shot      | 0.21952| 0.06218   | 0.15168| 0.15168 |
| Few-Shot | 0.27937| 0.06923| 0.16786| 0.17111 |

* **LangChain**

<img src = "https://github.com/convoluted-turtle/CSE-6242-Group-Project/assets/33863191/8abf78ae-5e67-4585-b77a-e3054b30cd09" alt = "LangChain Metrics" width = "700">
---

### Visualization Creation

#### Poster Display

#### Bar Chart

#### UMAP

![image](https://github.com/convoluted-turtle/CSE-6242-Group-Project/assets/33863191/4db9bea7-2dc4-458b-a09f-79cef207435f)


#### Box and Whiskers

---

### Application Creation 

#### Streamlit Application

<img src="https://github.com/convoluted-turtle/CSE-6242-Group-Project/assets/33863191/e7da3451-d5b8-4ef0-81cc-27bf73d4c503" alt="Streamlit Application" width="700">

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













