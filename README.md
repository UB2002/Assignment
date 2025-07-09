# Electronix AI - Binary Sentiment Analysis Microservice

An end-to-end microservice for binary sentiment analysis using a Hugging Face Transformer, with a React + Tailwind frontend and Docker Compose orchestration.

---

## 🔧 Features

* **REST API** for sentiment inference (`POST /predict`)
* **Fine-tuning script** via CLI (`finetune.py`)
* **React + TailwindCSS** frontend UI
* **Model hot-loading** from `./model` directory on startup
* **Fully Dockerized** services with Docker Compose
* **CPU-only** support (optional GPU)

---

## 🧠 Tech Stack

* **Backend**: Python, FastAPI, PyTorch, Transformers
* **Frontend**: React, TailwindCSS, Vite
* **Model**: `distilbert-base-uncased-finetuned-sst-2-english`
* **Containerization**: Docker, Docker Compose

---

## 🚀 Quickstart

### Prerequisites

* Docker & Docker Compose
* Node.js (for local frontend development)

---

### 1. Clone Repository

```bash
git clone https://github.com/your-username/electronix-ai.git
cd electronix-ai
```

### 2. Start Services

Bring up both backend and frontend:

```bash
docker-compose up --build
```

* **Backend** runs at: [http://localhost:3000](http://localhost:3000)
* **Frontend** runs at: [http://localhost:5173](http://localhost:5173)

---

### 3. Health Check

Verify backend is running:

```bash
curl http://localhost:3000/
# { "status": "ok" }
```

---

### 4. Prediction API

**Endpoint:** `POST /predict`

**Request Body:**

```json
{ "text": "I love this product!" }
```

**Response:**

```json
{
  "label": "positive",
  "score": 0.9876
}
```

---

### 5. Fine-Tuning Model

Prepare a `data.jsonl` file:

```jsonl
{"text": "Great service!", "label": "positive"}
{"text": "Terrible quality.", "label": "negative"}
```

Run CLI in container:

```bash
docker run --rm -v $PWD/backend:/app electronix-ai-backend \
  python finetune.py -data data.jsonl -epochs 3 -lr 3e-5
```

---

## 📦 Project Structure

```
electronix-ai/
├── backend/
│   ├── app/                # FastAPI application
│   ├── finetune.py         # Fine-tuning script
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/                # React source code
│   ├── index.html
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
└── README.md               # This file
```

---

