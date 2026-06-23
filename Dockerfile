FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir tensorflow flask pandas numpy pymysql requests

COPY . .

CMD ["python", "app.py"]
