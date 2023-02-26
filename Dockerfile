FROM python:3.9-slim

ENV APP_HOME /app
WORKDIR $APP_HOME

ADD requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY CLIP ./CLIP
COPY data.py main.py Search.py utils.py variables.py gcp_key.json ./

EXPOSE 80

# Run the web service on container startup using uvicorn webserver.
CMD exec uvicorn main:app --host 0.0.0.0 --port 80 --workers 1