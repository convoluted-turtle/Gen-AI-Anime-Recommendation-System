# CSE-6242-Group-Project
Gen-AI Anime Recommendation System

![image](https://github.com/convoluted-turtle/CSE-6242-Group-Project/assets/33863191/edfb1fa8-1288-4248-a59b-b91f60f5933a)

![image](https://github.com/convoluted-turtle/CSE-6242-Group-Project/assets/33863191/e92ba2d0-673f-4b07-b424-1670a6655ff8)

![image](https://github.com/convoluted-turtle/CSE-6242-Group-Project/assets/33863191/84fe2861-d853-4c19-84b8-228eaebaf56b)

![image](https://github.com/convoluted-turtle/CSE-6242-Group-Project/assets/33863191/cd6daed7-ca21-43d5-9730-e2a68197aacf)


Rebuild data image containers

```
docker build -f Dokcerfile-data -t jonlin8188/cse6242-data:latest .
docker push jonlin8188/cse6242-data:latest .
```

To run:

```
docker build -t jonlin8188/cse6242:latest .
docker run -p 8080:8080 --rm -it jonlin8188/cse6242:latest
# Access at http://localhost:8080
docker push jonlin8188/cse6242:latest
```