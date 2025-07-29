FROM python:3.10.18

RUN apt-get update && \
    apt-get install -y git libgl1-mesa-glx libglib2.0-0 && \
    python -m pip install --upgrade pip setuptools wheel && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app

EXPOSE 8000
CMD ["uvicorn", "detect-nid:app", "--host", "0.0.0.0", "--port", "8000"]