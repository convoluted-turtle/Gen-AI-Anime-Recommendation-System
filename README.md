# CSE-6242-Group-Project
## Gen-AI Anime Recommendation System

### Intro

<img src="https://github.com/convoluted-turtle/CSE-6242-Group-Project/assets/33863191/edfb1fa8-1288-4248-a59b-b91f60f5933a" alt="intro" width="700">


Our solution addresses the challenges faced by modern anime recommendation systems by introducing a hybrid approach enriched with visualizations. By integrating data from IMDB, Wikipedia, and MyAnimeList (MAL), we ensure comprehensive information on series and titles. We tackle cold start scenarios and long-tail issues by employing Item-KNN with a Jaccard similarity filter, guaranteeing diverse and relevant recommendations. Through UMAP visualization, users gain deeper insights into the anime landscape, while industry leaders and researchers benefit from a comprehensive view of consumer sentiment and trends. This approach not only enhances content discovery for fans but also provides a trusted source of truth in the anime community.

TLDR Steps:

1. We started by scraping anime datasets from IMDb and Wikipedia, combining them with the MyAnimeList dataset from Kaggle for comprehensive information.
    
    Web Scraping/API:
    * [IMDB WebScrape](web_scraping/imdb_pull_combine_my_anime.ipynb)
    * [Wiki API Text](web_scraping/wikipedia_text_pull.ipynb)
    * [Wiki API Image](web_scraping/wikipedia_image_scrape.ipynb)

    Data Join:
    * [IMDB Join](data_join/imdb_myanime_combine/imdb_data%20-2024-03-12-import.ipynb)
    * [Join Actor Names](data_join/imdb_myanime_combine/imdb_data%20-2024-03-13-createprincipalspivot.ipynb) 
    * [Image Join](web_scraping/wikipedia_image_join.ipynb)

2. Utilizing item-KNN collaborative filtering and pre-computed Thompson sampling for popular recommendations, we enhanced our recommendation algorithms.

    * [Item KNN-CF](recs/item-item-cf.ipynb)
    * [Thompson Sampling](recs/popular_recs.ipynb)

3. A vector database was constructed using data from Kaggle, IMDb, and Wikipedia, with the text columns from Wikipedia and MyAnimeList serving as embeddings.

    * [FAISS vectorDB](vector_database_creation/faiss_v_db.ipynb)

4. We crafted prompts to ground Gemini-Pro, imbuing it with extended knowledge derived from our comprehensive vector database.

    * [LangChain Prompting](prompting/guidance_prompting.ipynb) 
    * [Rouge Eval](prompting/prompting-eval.ipynb)
    * [LangChain Eval](prompting/langsmith-prompting-eval.ipynb)
    
5. Visualizations, including UMAP, were developed to showcase recommendations from the vector database, collaborative filtering, and popular suggestions, providing users with a holistic view.
6. We built a Streamlit application featuring various visualizations, with a chat option available for user interaction. Visualizations were predominantly focused on studio analysis in response to different user queries.
7. Lastly, the application was dockerized for seamless deployment and scalability.

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


To further elevate personalization based on users' favorite actors, actresses, or voice actors, we seamlessly integrated casting information from IMDb, enriching our dataset beyond what was available on MyAnimeList (MAL) from Kaggle. By web scraping IMDb, we ensure comprehensive insights into the talent behind each anime, enhancing the recommendation process. This augmentation not only caters to individual preferences but also adds depth to the user experience, aligning with our commitment to providing rich, diverse, and personalized recommendations in the ever-evolving anime landscape.

IMDB Scraping Notebook: [here](web_scraping/imdb_pull_combine_my_anime.ipynb)

#### Wikipedia Plot API

In addressing issues with plot information or anime descriptions, we integrated the Wikipedia API to gather comprehensive summaries and plot details from each anime's Wikipedia page. This dynamic approach ensures that our recommendation system presents accurate and detailed information, enhancing the user experience. By leveraging this additional data source, we enrich our dataset with comprehensive insights, further refining our ability to provide rich, diverse, and personalized recommendations. This fluid integration enables us to overcome limitations and deliver a more comprehensive understanding of each anime, fostering greater engagement and satisfaction among users.

WikiPedia Text API Notebook: [here](web_scraping/wikipedia_text_pull.ipynb)

#### Wikipedia Image API

In addressing broken anime image links, we merged our dataset with MyAnimeList, combining their image with WikiPedia Images. This integration resolves issues of missing or broken image links, ensuring a visually enriched experience for users.

Wikipedia Image API Notebook: [here](web_scraping/wikipedia_image_scrape.ipynb)


---

### Data Joining

#### Joining IMDB


One of the primary challenges we encountered was matching similar anime titles, such as variations like "Dragon Ball 1, 2, 3, 4," which lacked a common identifier for matching between the IMDb and MyAnimeList datasets. Additionally, language differences posed another hurdle, as some titles were presented in Romanization or Katakana, while Wikipedia often provided the Westernized names. After the join we went with the anime_id as the identfier. 

Databricks importing Kaggle and IMDB dataset to join Notebook: [here](data_join/imdb_myanime_combine/imdb_data%20-2024-03-12-import.ipynb)

Joining in Actor Names and Pivoting Notebook: [here](data_join/imdb_myanime_combine/imdb_data%20-2024-03-13-createprincipalspivot.ipynb) 

MyAnimeList Dataset link: https://www.kaggle.com/datasets/azathoth42/myanimelist

#### Joining Wikipedia

Images were then further joined back in for any broken links.   Another round was done to try to add more selection of anime to the dataset which pushed the number of recommended content to about 8000. 

Wikipedia Image Join to Data Notebook: [here](web_scraping/wikipedia_image_join.ipynb)

---

### Collobroative Filtering and Popularity

#### Item-KNN Collabroative Filtering


To utilize insights from the click history dataset sourced from MyAnimeList on Kaggle, we implemented an item-to-item collaborative filtering. This technique utilizes adjusted cosine similarity to get anime titles based on user click behavior. By leveraging this approach, we effectively capture user preferences and recommend similar anime titles, enriching the recommendation process with personalized and relevant suggestions. 

Item KNN-CF Notebook: [here](recs/item-item-cf.ipynb)

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

In our recommendation system, we employed Thompson sampling as it offers significant advantages over simply relying on frequency-based methods. Unlike frequency-based approaches, which prioritize items based solely on their popularity or occurrence frequency, Thompson sampling dynamically balances exploration and exploitation. By leveraging probabilistic sampling, Thompson sampling allows our system to continually explore lesser-known anime titles while exploiting the knowledge gained from user interactions. This adaptive strategy not only promotes diversity in recommendations but also maximizes long-term user engagement by continually learning and adapting to evolving preferences and trends. 

Popular Notebook: [here](recs/popular_recs.ipynb)

---

### Vector DataBase Creation 

#### FAISS Vector Database Creation

To enhance our recommendation system further, we integrated Thompson sampling and item-to-item collaborative filtering, into a unified framework using a vector database powered by FAISS. This would boost the diversity and range of recommendations we could recommend to the user. 

Additionally, we incorporated a metadata filtering that takes into account various factors such as genre, actors, producers, ratings, and studios. Furthermore, our vector database employs a hybrid search approach, combining keyword-based searches with dense vector cosine similarity for efficient document retrieval. 

![image](https://github.com/convoluted-turtle/CSE-6242-Group-Project/assets/33863191/e92ba2d0-673f-4b07-b424-1670a6655ff8)

FAISS Vector DB Notebook: [here](vector_database_creation/faiss_v_db.ipynb)

#### Filter Creation


This filter function is designed to sift through retrieved documents based on their metadata attributes. It begins by extracting relevant metadata tokens, such as studio, producer, licensors, and genre, from each document. The function then iterates through a list of query tokens, checking if any of them match with the metadata tokens. Additionally, it considers documents with a score higher than a rating score a user might prefer to pass the filter. If any metadata attribute matches any query token, the document is deemed to pass the filter. 

---

### Prompting


We employed Langchain to orchestrate our retrieval-augmented generation pipeline, streamlining the process of generating personalized recommendations. Using a four-shot prompt, we infused the chatbot with the persona of an avid anime lover, ensuring that recommendations are tailored to the tastes and preferences of anime enthusiasts. This persona-based approach enables the chatbot to engage with users on a more personal level, understanding their unique interests and providing relevant suggestions accordingly. 

LangChain Prompting Notebook: [here](prompting/guidance_prompting.ipynb) 

Guidance Prompting Notebook: [here](prompting/langchain_prompting.ipynb)

LLM used Gemini Pro: https://gemini.google.com

#### Prompt Template Creation

<img src="https://github.com/convoluted-turtle/CSE-6242-Group-Project/assets/33863191/a046c012-2bb7-46d5-951f-62e5c1dbba65" alt="prompting" width="600">


#### Prompting Evaluation


In our evaluation process, we utilized Rouge scores, a set of metrics commonly employed in natural language processing tasks to assess the quality of generated text against reference summaries or ground truth. Rouge-1 measures the overlap of unigram (single word) sequences between the generated text and the reference summary. Rouge-2 extends this to measure the overlap of bigram (two-word) sequences. Rouge-L computes the longest common subsequence between the generated text and the reference summary, considering the length of the longest common subsequence. Rouge-Lsum evaluates the average Rouge-L score across multiple reference summaries. 

* **Rouge**

| Prompt/Metric                | Rouge1  | Rouge2 | RougeL   | RougeLsum |
|----------------------|-------|----------|-------|--------|
| Zero-Shot       | 0.22676| 0.07376  | 0.15299| 0.18761 |
| One-Shot      | 0.21952| 0.06218   | 0.15168| 0.15168 |
| Few-Shot | 0.27937| 0.06923| 0.16786| 0.17111 |

Rouge Notebook: [here](prompting/prompting-eval.ipynb)

* **LangSmith**

Additionally, we employed a Chain of Thought approach using Langsmith to trace back the generation steps of our chat model, Gemini-Pro. This method enables a meticulous analysis of each response, ensuring logical coherence and alignment with the initial question. This offline evaluation, conducted with a ground-truth dataset curated by our team, provides valuable insights into the effectiveness and accuracy of our recommendation system's responses.

<img src = "https://github.com/convoluted-turtle/CSE-6242-Group-Project/assets/33863191/8abf78ae-5e67-4585-b77a-e3054b30cd09" alt = "LangChain Metrics" width = "700">

LangSmith Prompting Eval Notebook: [here](prompting/langsmith-prompting-eval.ipynb)

---

### Visualization Creation

In our visualization setup, we employed a dynamic combination of graphical elements to present insights from our recommendation system effectively. Utilizing a bar chart, we showcased the top-rated anime selected by the user's query, offering a clear depiction of popular choices. Box and whisker plots were used to illustrate user favorites across different studios, providing a comprehensive view of studio performance. Additionally, three anime posters, representing the closest matches to the query, offered a visually engaging way to explore recommended titles. Alongside these visualizations, a chat window facilitated real-time interaction with a chatbot, enriching the user experience with personalized responses and recommendations. Positioned at the bottom, a UMAP visualization mapped the positions of item-KNN collaborative filtering, popular recommendations, and vector database recommendations, offering deeper insights into the recommendation landscape.

#### UMAP


In our Uniform Manifold Approximation and Projection (UMAP) visual, we carefully select columns such as genre, synopsis, air date, actors' names, studios, producers, and anime score to capture diverse aspects of each anime. UMAP is a dimensionality reduction technique that preserves local and global structure in high-dimensional data, allowing us to visualize the relationships between anime titles in a lower-dimensional space. By leveraging UMAP, we can uncover complex patterns and similarities between anime titles that might not be apparent in the original high-dimensional space.

To enhance recommendation quality, we concatenate embeddings generated from nearest neighbors of 5 using Euclidean distance across different recommendation techniques, including the vector database, item-KNN collaborative filtering, and popular recommendations. This approach enables us to capture nuanced relationships between anime titles based on various factors, fostering a more personalized recommendation experience for users.

Once recommendations are collected, they can be utilized in marketing emails or other communications to encourage users to engage with more content on the site. By tailoring recommendations to individual preferences and interests, we not only promote user engagement but also enhance user satisfaction and retention. 

![image](https://github.com/convoluted-turtle/CSE-6242-Group-Project/assets/33863191/4db9bea7-2dc4-458b-a09f-79cef207435f)

---

### Application Creation 

This application serves dual purposes: on the left side, it provides customers with an intuitive chat interface for browsing our extensive anime database, enabling them to ask questions and receive personalized recommendations efficiently, enhancing their viewing experience. Meanwhile, on the right side, studio analysts delve into user queries, gaining insights into search patterns and preferences. This analysis empowers them to tailor promotions via email or text to boost engagement and identify trends for potential anime productions. Visual aids, including a bar chart showcasing anime ratings and a box and whisker plot depicting user favorite frequencies, offer valuable insights. The poster display aids studio executives in visually assessing recommended anime relative to the query, facilitating potential redesign considerations. Looking forward, image embeddings will further enrich the user experience. The inclusion of UMAP visualization, integrating various recommendation algorithms, enables the studio to discern the effectiveness of recommendations and optimize promotional strategies for increased traffic.

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













