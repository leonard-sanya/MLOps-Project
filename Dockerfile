FROM python:3.11.5

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

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN pip install uvicorn
RUN pip list

COPY ./main.py /code/main.py


EXPOSE 8080


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]



