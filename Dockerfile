# Use an official Python runtime as a parent image
# FROM python:3.9-slim

# Set the working directory in the container
# WORKDIR /app

# # Install system dependencies for dlib
# RUN apt-get update && \
#     apt-get install -y build-essential cmake && \
#     apt-get install -y libgtk-3-dev && \
#     apt-get install -y libboost-python-dev && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# # Copy the requirements file to the container
# COPY requirements.txt .
# RUN pip install uvicorn
# # Install the dependencies from the requirements file
# RUN /bin/sh -c pip install --no-cache-dir --upgrade -r requirements.tx
# # Copy the rest of your application code to the container
# COPY main.py app/main.py

# # Expose port 80 for the app
# EXPOSE 80

# # Start the app with Uvicorn
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

# Use the official Python image from the Docker Hub
FROM python:3.9

# Set the working directory in the container
WORKDIR /app
RUN apt-get update && \
    apt-get install -y build-essential cmake && \
    apt-get install -y libgtk-3-dev && \
    apt-get install -y libboost-python-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Install the Python dependencies
RUN /bin/sh -c pip install --no-cache-dir --upgrade -r requirements.tx

# Copy the rest of the app code to the working directory
COPY . .

# Expose port 80 to the outside world
EXPOSE 80

# Command to run the FastAPI app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
