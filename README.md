# CSE-6242-Group-Project
## Gen-AI Anime Recommendation System

### Dependencies 

| pypi library | version |
|----------|----------|
| Cell 1   | Cell 2   |
| Cell 3   | Cell 4   |
| Cell 5   | Cell 6   |
| Cell 7   | Cell 8   |
| Cell 9   | Cell 10  |
| Cell 11  | Cell 12  |
| Cell 13  | Cell 14  |
| Cell 15  | Cell 16  |

### Intro

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

![image](https://github.com/convoluted-turtle/CSE-6242-Group-Project/assets/33863191/edfb1fa8-1288-4248-a59b-b91f60f5933a)

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













