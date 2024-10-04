# Use a python image as the base
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev

# Set the working directory
WORKDIR /code

# Copy the requirements to the working directory
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the app directory contents to the working directory
COPY ./main.py /code/main.py

EXPOSE 80
# Run the FastAPI application using Uvicorn
CMD ["fastapi", "run", "main.py", "--port", "80"]
