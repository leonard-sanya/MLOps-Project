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

# Specify the command to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
