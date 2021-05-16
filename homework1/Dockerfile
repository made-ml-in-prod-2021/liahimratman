FROM python:3.7

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ml_project ml_project
COPY configs configs
COPY online_inference online_inference
COPY api.py api.py

WORKDIR .

CMD uvicorn api:app --host=0.0.0.0 --port=80