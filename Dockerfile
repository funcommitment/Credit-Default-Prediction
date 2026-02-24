FROM python:3.12.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "streamlit run Streamlitappcredit/app.py --server.address=0.0.0.0 --server.port=${PORT:-8080}"]

