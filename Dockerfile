FROM python:3.10.0
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt ./

RUN apt-get update && apt-get install -y \
    python3-pip 
    

RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0



RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "server.py"]