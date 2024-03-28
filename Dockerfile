# # Clone the embedding model and move it into 
# FROM python:3.11-slim-bullseye as model
# RUN apt-get update && apt-get install -y git
# RUN git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 /tmp/model
# RUN rm -rf /tmp/model/.git

FROM jonlin8188/cse6242-data:latest as data

# Build the streamlit container
FROM python:3.11-slim-bullseye
ENV HOST=0.0.0.0 
ENV LISTEN_PORT 8080
EXPOSE 8080
RUN apt-get update && apt-get install -y git build-essential
COPY ./app/requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir --upgrade -r /app/requirements.txt --default-timeout=1000 --no-cache-dir
RUN python -m spacy download en_core_web_sm
WORKDIR /app
 
# Huggingface embedding model
# COPY --from=model /tmp/model /app/embedding_model

# Datasets
COPY --from=data /data/ /app/data
COPY --from=data /model/ /app/model
COPY --from=data /embedding/ /app/embedding

# Copy streamlit assets
COPY ./app /app

# https://fgiasson.com/blog/index.php/2023/08/23/how-to-deploy-hugging-face-models-in-a-docker-container/

CMD ["streamlit", "run", "anime_app.py", "--server.port", "8080"]
