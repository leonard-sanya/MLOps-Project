FROM python
 # :3.9

# Set the working directory inside the container
WORKDIR /code

# Install system dependencies needed by OpenCV

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Copy the requirements file into the working directory
COPY ./requirements.txt /code/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the application code into the working directory
COPY ./main.py /code/main.py

# Copy the models directory into the working directory
COPY ./models /code/models

EXPOSE 8000
# ENV MLFLOW_TRACKING_USERNAME=ignatiusboadi
# ENV MLFLOW_TRACKING_PASSWORD=67ea7e8b48b9a51dd1748b8bb71906cc5806eb09
ENV MLFLOW_TRACKING_URI=https://dagshub.com/ignatiusboadi/mlops-tasks.mlflow
ENV MLFLOW_EXPERIMENT_NAME=Celebrity-face-recognition


# Specify the command to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
