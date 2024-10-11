FROM python:3.11.5

# Install system dependencies
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    fswebcam \
    v4l-utils \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /code

# Copy the requirements to the working directory
COPY ./requirements.txt /code/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Install uvicorn separately (if not already in requirements.txt)
RUN pip install uvicorn
RUN pip list
# Copy the app directory contents to the working directory
COPY ./main.py /code/main.py

# Expose the port (8000 for FastAPI)
EXPOSE 8080

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]



