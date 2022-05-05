FROM python:3.8
COPY requirements.txt .
RUN pip install -r requirements.txt && python -m nltk.downloader stopwords && python -m nltk.downloader wordnet && python -m nltk.downloader punkt
WORKDIR /topic_modeling
COPY FlaskApplication .
RUN mkdir -p logs
RUN ["chmod", "+x", "gunicorn.sh"]
EXPOSE 5000
ENTRYPOINT ["./gunicorn.sh"]