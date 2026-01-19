# 🎤 AI Meeting Assistant — Speech to Text, Summary & Translation

A full-stack AI meeting assistant that records audio from the browser, transcribes speech into text using Whisper, summarizes using Ollama LLM, and translates into multiple languages — all running locally with **free & open-source tools**.

Designed for **CPU-only systems** (Intel i5 / Iris Xe supported).

---

## ✨ Features

* 🎙 Browser audio recording (WAV high quality)
* 📝 Speech-to-text using OpenAI Whisper (offline)
* 🌍 Automatic language detection
* 🤖 AI summarization using Ollama (LLaMA model)
* 🌐 Translation (Malayalam, Hindi, Tamil, English, etc.)
* 📁 Automatic audio storage
* ⚡ Runs fully offline after model download
* 💻 Optimized for CPU (no GPU required)

---

## 🧠 Tech Stack

### Backend

* **Python 3.11**
* **FastAPI** — API server
* **Whisper (OpenAI)** — Speech recognition
* **Torch (CPU)** — ML runtime
* **Soundfile + Librosa** — Audio processing
* **Ollama** — AI summarization
* **Deep Translator** — Translation
* **Uvicorn** — ASGI server

### Frontend

* **HTML + JavaScript**
* **Web Audio API**
* **Fetch API**

---

## 📁 Project Structure

```
ai-meeting-assistant/
│
├── backend-python/
│   ├── main.py              # FastAPI backend
│   ├── speech.py            # Whisper speech-to-text
│   ├── summarize.py         # Ollama summarization
│   ├── translate.py         # Translation
│   ├── requirements.txt    # Python dependencies
│   ├── recordings/         # Auto-created audio folder
│
├── frontend/
│   └── index.html           # Web UI
│
└── README.md
```

The `recordings/` folder is automatically created when the backend starts.

---

## 🖥 System Requirements

* Windows / Linux / macOS
* Python 3.10+
* 8GB RAM minimum (16GB recommended)
* No GPU required
* 10GB free disk (for models)

---

## 🚀 Installation Guide

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/ai-meeting-assistant.git
cd ai-meeting-assistant/backend-python
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Install Ollama

Download from:

```
https://ollama.com
```

Then install LLaMA model:

```bash
ollama pull llama3.2:1b
```

Verify:

```bash
ollama run llama3.2:1b
```

---

### 5️⃣ Start Backend

```bash
uvicorn main:app --reload --port 8000
```

Backend will start at:

```
http://127.0.0.1:8000
```

---

### 6️⃣ Start Frontend

Open `frontend/index.html` using Live Server or:

```bash
python -m http.server 5500
```

Then open:

```
http://localhost:5500
```

---

## 📥 Whisper Model Download

On first run, Whisper automatically downloads the model:

```python
whisper.load_model("medium", device="cpu")
```

Models are stored in:

```
C:\Users\USERNAME\.cache\whisper
```

Available models:

| Model  | Speed  | Accuracy |
| ------ | ------ | -------- |
| small  | Fast   | Medium   |
| medium | Medium | High     |
| large  | Slow   | Best     |

Recommended: **medium**

---

## 🔁 How It Works

```
Mic → Browser → WAV → FastAPI → Whisper → Text → Ollama → Summary → Translation
```

Whisper loads only once on server startup and is reused for all recordings.

---

## 🌍 Supported Languages

* Malayalam
* Hindi
* Tamil
* English
* Spanish
* French
* German
* Chinese
* Japanese
* Korean

---

## 📦 API Endpoints

| Endpoint             | Description      |
| -------------------- | ---------------- |
| `/api/process-audio` | Speech-to-text   |
| `/api/summarize`     | AI summary       |
| `/api/translate`     | Translation      |
| `/api/recordings`    | List saved audio |
| `/api/health`        | Server status    |

---

## ⚡ Performance

Optimized for CPU:

* Uses mono 16kHz audio
* Model cached in memory
* No reload per request
* No ffmpeg required

---

## 🔐 Privacy

* All processing is local
* No cloud APIs
* No audio leaves your machine

---

