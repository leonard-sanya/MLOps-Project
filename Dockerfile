FROM python

WORKDIR /code


COPY ./requirements.txt /code/requirements.txt


RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


COPY ./main.py /code/main.py


# COPY ./models /code/models

EXPOSE 8001

# ENV MLFLOW_TRACKING_URI=https://dagshub.com/ignatiusboadi/mlops-tasks.mlflow
# ENV MLFLOW_EXPERIMENT_NAME=Celebrity-face-recognition


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
