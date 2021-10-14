FROM tiangolo/uvicorn-gunicorn:python3.8-slim

WORKDIR /app

ADD requirements.txt .
RUN pip install -r requirements.txt

#COPY ./model_deployment /model_deployment
#COPY ./app.py .
#COPY ./autonlp /autonlp
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]

#docker build -t autonlp .
#docker run -d --name apiautonlpcontainer -p 5000:5000 autonlp
#docker run -p 5000:5000 autonlp