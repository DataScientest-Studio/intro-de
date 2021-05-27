# MC Introduction to Data Engineering

This repo contains code related to the "Intro to Data Engineering" masterclass.

Mainly, it showcases how we can develop an HTTP API using [Flask](https://flask.palletsprojects.com/en/2.0.x/) or [FastAPI](https://fastapi.tiangolo.com/) to deploy a machine learning model trained in a jupyter notebook.

The relevant files are :

- [notebooks/training.ipynb](./notebooks/training.ipynb) : notebook used to train the ML model
- [api/api_flask.py](./api/api_flask.py) : Flask API to deploy the model
- [api/api_fastapi.py](./api/api_fastapi.py) : FastAPI API to deploy the model


To run the Flask API :

```python
cd api 
python app.py
```

To run the FastAPI API :

```python
cd api 
uvicorn api_fastapi:api --reload
```
