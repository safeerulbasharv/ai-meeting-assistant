from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uuid
import shutil
import socket
import platform
from datetime import datetime
import time
import traceback

from summarize import summarize_text, check_ollama_status, test_ollama
from speech import transcribe_audio
from translate import translate_text

app = FastAPI(title="Speech-to-Text Backend", version="2.0")

# ========== CORS CONFIGURATION ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500", 
        "http://127.0.0.1:5500",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== DIRECTORY SETUP ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECORD_DIR = os.path.join(BASE_DIR, "recordings")
os.makedirs(RECORD_DIR, exist_ok=True)

def get_local_ip():
    """Get local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


# === Ollama Integration Endpoints ===

@app.get("/api/ollama/status")
async def ollama_status():
    """Check Ollama service status"""
    try:
        status = check_ollama_status()
        return {
            "success": True,
            "ollama_available": status["ollama_available"],
            "model": status["model"],
            "base_url": status["base_url"],
            "timestamp": datetime.now().isoformat(),
            "message": "Ollama service status"
        }
    except Exception as e:
        return {
            "success": False,
            "ollama_available": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/ollama/summarize")
async def ollama_summarize(data: dict):
    """Summarize text using Ollama Llama model"""
    try:
        text = data.get("text", "")
        language = data.get("language", "en")
        use_ollama = data.get("use_ollama", True)
        
        if not text or len(text.strip()) < 10:
            raise HTTPException(status_code=400, detail="Text too short for summarization")
        
        print(f"🤖 Ollama summarization request")
        print(f"📊 Text length: {len(text)} chars")
        print(f"🌍 Language: {language}")
        
        # Use Ollama for summarization
        summary = summarize_text(text, language, use_ollama)
        
        return {
            "success": True,
            "summary": summary,
            "originalLength": len(text),
            "summaryLength": len(summary),
            "compression": f"{len(summary)/len(text)*100:.1f}%" if text else "0%",
            "engine": "ollama" if use_ollama and not summary.startswith("[Error:") else "fallback",
            "language": language,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Ollama summarize error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Ollama summarization failed: {str(e)}")

@app.post("/api/ollama/test")
async def ollama_test():
    """Test Ollama connection and summarization"""
    try:
        print("🧪 Testing Ollama integration...")
        result = test_ollama()
        
        return {
            "success": result,
            "message": "Ollama test completed",
            "ollama_available": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"❌ Ollama test error: {e}")
        return {
            "success": False,
            "message": f"Ollama test failed: {str(e)}",
            "ollama_available": False,
            "timestamp": datetime.now().isoformat()
        }


# ========== STARTUP EVENT ==========
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("=" * 70)
    print("🚀 SPEECH-TO-TEXT BACKEND STARTING")
    print("=" * 70)
    
    local_ip = get_local_ip()
    
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"📁 Recordings directory: {RECORD_DIR}")
    print(f"💻 Hostname: {socket.gethostname()}")
    print(f"🌐 Local IP: {local_ip}")
    print(f"🐍 Python: {platform.python_version()}")
    print(f"🖥️  Platform: {platform.platform()}")
    
    # Test directory permissions
    try:
        test_file = os.path.join(RECORD_DIR, f"test_{int(time.time())}.txt")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print("✅ Recordings directory is writable")
    except Exception as e:
        print(f"❌ Directory error: {e}")
    
    print("\n🔧 Testing imports...")
    imports_to_test = [
        ("whisper", "openai-whisper"),
        ("langdetect", "langdetect"),
        ("deep_translator", "deep-translator"),
        ("torch", "torch"),
        ("numpy", "numpy"),
    ]
    
    for import_name, package_name in imports_to_test:
        try:
            __import__(import_name)
            print(f"  ✅ {package_name}")
        except ImportError as e:
            print(f"  ❌ {package_name}: {e}")
    
    print("\n" + "=" * 70)
    print(f"🌐 BACKEND URL: http://127.0.0.1:8000")
    print(f"🌐 ALTERNATE URL: http://{local_ip}:8000")
    print(f"🔗 FRONTEND URL: http://localhost:5500")
    print(f"📚 API DOCS: http://127.0.0.1:8000/docs")
    print(f"🏥 HEALTH CHECK: http://127.0.0.1:8000/api/health")
    print("=" * 70)
    print("✅ BACKEND READY - Waiting for requests...")
    print("=" * 70)

# ========== REQUEST LOGGING MIDDLEWARE ==========
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Skip logging for frequent endpoints
    skip_logging = ["/favicon.ico", "/docs", "/openapi.json"]
    if request.url.path not in skip_logging:
        client_ip = request.client.host if request.client else "unknown"
        print(f"📨 {request.method} {request.url.path} from {client_ip}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        if request.url.path not in skip_logging:
            print(f"✅ {request.method} {request.url.path} - {response.status_code} ({process_time:.2f}s)")
        
        return response
    except Exception as e:
        process_time = time.time() - start_time
        print(f"❌ {request.method} {request.url.path} - ERROR ({process_time:.2f}s): {e}")
        raise

# ========== API ENDPOINTS ==========

@app.get("/")
async def root():
    """Root endpoint with API documentation"""
    return {
        "message": "🎤 Speech-to-Text Backend API",
        "version": "2.0",
        "status": "running",
        "endpoints": {
            "GET /": "This documentation",
            "GET /api/health": "Health check",
            "GET /api/config": "Frontend configuration",
            "GET /api/test": "Connection test",
            "GET /api/ping": "Simple ping",
            "GET /api/recordings": "List audio files",
            "POST /api/process-audio": "Process audio in real-time",
            "POST /api/summarize": "Summarize text",
            "POST /api/translate": "Translate text"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/health")
async def health_check(request: Request):
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "speech-to-text",
        "timestamp": datetime.now().isoformat(),
        "client_ip": request.client.host if request.client else "unknown",
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "uptime": "Just started"  # Add uptime tracking if needed
    }

@app.get("/api/config")
async def config(request: Request):
    """Frontend configuration"""
    local_ip = get_local_ip()
    
    return {
        "baseUrl": "http://127.0.0.1:8000",
        "localIpUrl": f"http://{local_ip}:8000",
        "supportedFormats": [".webm", ".wav", ".mp3", ".ogg"],
        "maxFileSize": 50 * 1024 * 1024,  # 50MB
        "serverInfo": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "timestamp": datetime.now().isoformat()
        },
        "features": {
            "recording": True,
            "transcription": True,
            "language_detection": True,
            "summarization": True,
            "translation": True,
            "supported_languages": ["ml", "hi", "en", "es", "fr", "ta"]
        }
    }

@app.get("/api/test")
async def test_endpoint(request: Request):
    """Connection test endpoint"""
    return {
        "success": True,
        "message": "✅ Backend is reachable!",
        "timestamp": datetime.now().isoformat(),
        "client_ip": request.client.host if request.client else "unknown",
        "backend_url": "http://127.0.0.1:8000",
        "frontend_url": "http://localhost:5500",
        "status": "operational"
    }

@app.get("/api/ping")
async def ping():
    """Simple ping endpoint"""
    return {
        "status": "pong",
        "timestamp": datetime.now().isoformat(),
        "message": "Backend is alive!"
    }

def save_audio_file(upload: UploadFile, max_size_mb: int = 50):
    """Save uploaded audio file with validation"""
    if not upload.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_extension = os.path.splitext(upload.filename)[1].lower()
    
    allowed_extensions = ['.webm', '.wav', '.mp3', '.ogg', '.m4a']
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format: {file_extension}. Allowed: {', '.join(allowed_extensions)}"
        )
    
    filename = f"{uuid.uuid4()}{file_extension}"
    path = os.path.join(RECORD_DIR, filename)
    
    try:
        contents = upload.file.read()
        file_size = len(contents)
        
        if file_size > max_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large: {file_size/1024/1024:.1f}MB. Max {max_size_mb}MB"
            )
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        with open(path, "wb") as f:
            f.write(contents)
        
        print(f"✅ Saved: {filename} ({file_size/1024:.1f} KB)")
        return path
        
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(path):
            os.remove(path)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    finally:
        upload.file.close()

# === Audio Processing Endpoints ===

@app.post("/api/process-audio")
async def process_audio(audio: UploadFile = File(...)):
    try:
        print(f"🎤 Processing audio: {audio.filename}")
        path = save_audio_file(audio)

        result = transcribe_audio(path)
        text = result["text"]
        lang = result["language"]

        return JSONResponse(
            content={
                "success": True,
                "transcription": text,   # ✅ correct key
                "language": lang,
                "confidence": 95,
                "file": os.path.basename(path),
                "timestamp": datetime.now().isoformat()
            },
        )

    except Exception as e:
        print(f"❌ process-audio error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))



# === Text Processing Endpoints ===

@app.post("/api/summarize")
async def summarize_endpoint(data: dict):
    """Summarize text"""
    try:
        text = data.get("text", "")
        if not text or len(text.strip()) < 10:
            raise HTTPException(status_code=400, detail="Text too short for summarization")
        
        print(f"📊 Summarizing text ({len(text)} chars)")
        summary = summarize_text(text)
        
        return {
            "success": True,
            "summary": summary,
            "originalLength": len(text),
            "summaryLength": len(summary),
            "compression": f"{len(summary)/len(text)*100:.1f}%" if text else "0%",
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ summarize error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@app.post("/api/translate")
async def translate_endpoint(data: dict):
    """Translate text to target language"""
    try:
        text = data.get("text", "")
        target_lang = data.get("targetLang", "en")
        
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="No text provided for translation")
        
        valid_languages = ['ml', 'hi', 'en', 'es', 'fr', 'ta', 'de', 'zh', 'ja', 'ko']
        if target_lang not in valid_languages:
            target_lang = 'en'
        
        print(f"🌐 Translating to {target_lang} ({len(text)} chars)")
        translated = translate_text(text, target_lang)
        
        return {
            "success": True,
            "translation": translated,
            "targetLanguage": target_lang,
            "original": text[:100] + "..." if len(text) > 100 else text,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ translate error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

# === Utility Endpoint ===

@app.get("/api/recordings")
async def list_recordings():
    """List all recorded audio files"""
    files = []
    for filename in os.listdir(RECORD_DIR):
        filepath = os.path.join(RECORD_DIR, filename)
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath)
            created = os.path.getctime(filepath)
            files.append({
                "name": filename,
                "size": size,
                "size_human": f"{size/1024:.1f} KB",
                "created": datetime.fromtimestamp(created).isoformat(),
                "extension": os.path.splitext(filename)[1].lower()
            })
    
    return {
        "success": True,
        "count": len(files),
        "recordings": sorted(files, key=lambda x: x["created"], reverse=True),
        "timestamp": datetime.now().isoformat()
    }

# ========== ERROR HANDLERS ==========

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "message": "Request failed",
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    print(f"❌ Unhandled error: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "message": "Something went wrong",
            "timestamp": datetime.now().isoformat()
        }
    )

# ========== RUN SERVER ==========
if __name__ == "__main__":
    import uvicorn
    print("Starting server with uvicorn...")
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )