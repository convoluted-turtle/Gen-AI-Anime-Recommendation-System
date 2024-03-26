# Clone the embedding model and move it into 
FROM python:3.11-slim-bullseye as model
RUN apt-get update && apt-get install -y git
RUN git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 /tmp/model
RUN rm -rf /tmp/model/.git

# Build the streamlit container
FROM python:3.11-slim-bullseye
ENV HOST=0.0.0.0 
ENV LISTEN_PORT 8080
EXPOSE 8080
RUN apt-get update && apt-get install -y git build-essential
COPY ./requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir --upgrade -r /app/requirements.txt --default-timeout=1000 --no-cache-dir
WORKDIR /app
 
# TODO project repo - move app into app folder along with assets

# Temporary placeholder
COPY ./2024-03-18_22-38-08.png /app/2024-03-18_22-38-08.png

# Streamlit config
COPY ./streamlit /app/.streamlit

# Actual app
COPY ./anime_app.py /app/anime_app.py

# Huggingface embedding model
COPY --from=model /tmp/model /app/embedding_model

# https://fgiasson.com/blog/index.php/2023/08/23/how-to-deploy-hugging-face-models-in-a-docker-container/

CMD ["streamlit", "run", "anime_app.py", "--server.port", "8080"]
