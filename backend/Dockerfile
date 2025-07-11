FROM python:3.12-slim


WORKDIR /app


COPY requirements.txt .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt


COPY app/ ./app/
COPY finetune.py .

RUN mkdir -p /app/model

EXPOSE 3000

# running the backend

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3000"]
