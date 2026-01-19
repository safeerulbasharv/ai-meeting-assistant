import requests
import json
import time
import logging
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaSummarizer:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:1b"):
        self.base_url = base_url
        self.model = model
        self.timeout = 220  # 2 minutes timeout for longer summaries
        
    def check_connection(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                logger.info(f"✅ Ollama connected. Available models: {[m.get('name') for m in models]}")
                return True
        except requests.exceptions.ConnectionError:
            logger.warning("❌ Ollama not running. Please start Ollama with: ollama serve")
        except Exception as e:
            logger.error(f"❌ Ollama connection error: {e}")
        return False
    
    def summarize_with_ollama(self, text: str, language: str = "en") -> str:
        """Summarize text using Ollama Llama model"""
        print(f"\n🤖 OLLAMA SUMMARIZATION")
        print(f"📊 Input length: {len(text)} characters")
        print(f"🌍 Language: {language}")
        print(f"🤖 Model: {self.model}")
        
        if not self.check_connection():
            return "[Error: Ollama not running. Please start with: ollama serve]"
        
        if not text or len(text.strip()) < 50:
            return text
        
        try:
            # Prepare prompt based on language
            if language == "ml":  # Malayalam
                prompt = f"""ഇനിപ്പറയുന്ന വാചകം സംഗ്രഹിക്കുക. സംഗ്രഹം മലയാളത്തിൽ തന്നെ ആയിരിക്കണം:

{text[:3000]}

സംഗ്രഹം:"""
            elif language == "hi":  # Hindi
                prompt = f"""निम्नलिखित पाठ का सारांश दें। सारांश हिंदी में ही होना चाहिए:

{text[:3000]}

सारांश:"""
            elif language == "ta":  # Tamil
                prompt = f"""பின்வரும் உரையை சுருக்கவும். சுருக்கம் தமிழிலேயே இருக்க வேண்டும்:

{text[:3000]}

சுருக்கம்:"""
            else:  # English (default)
                prompt = f"""Please summarize the following text. Keep the summary concise (3-5 sentences). Focus on key points and main ideas:

{text[:3000]}

Summary:"""
            
            # Limit text length for API
            max_chars = 4000
            if len(text) > max_chars:
                print(f"⚠️  Truncating text from {len(text)} to {max_chars} characters")
                text = text[:max_chars] + "..."
            
            # Prepare request
            request_data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for more focused summaries
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": 500,  # Max tokens for summary
                    "stop": ["\n\n", "Summary:", "सारांश:", "സംഗ്രഹം:", "சுருக்கம்:"]
                }
            }
            
            print("⏳ Generating summary with Llama 3.2...")
            start_time = time.time()
            
            # Call Ollama API
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return f"[Error: {error_msg[:100]}]"
            
            result = response.json()
            summary = result.get("response", "").strip()
            
            process_time = time.time() - start_time
            print(f"✅ Summary generated ({process_time:.1f}s)")
            
            # Clean up the summary
            summary = self.clean_summary(summary, language)
            
            print(f"📊 Summary length: {len(summary)} characters")
            print(f"📄 Preview: {summary[:100]}...")
            
            return summary
            
        except requests.exceptions.Timeout:
            error_msg = "Ollama request timeout. Model might be too slow."
            logger.error(error_msg)
            return f"[Error: {error_msg}]"
        except Exception as e:
            error_msg = f"Ollama summarization error: {str(e)}"
            logger.error(error_msg)
            return f"[Error: {error_msg[:100]}]"
    
    def clean_summary(self, summary: str, language: str) -> str:
        """Clean up the summary output"""
        # Remove unnecessary prefixes
        prefixes = ["Summary:", "सारांश:", "സംഗ്രഹം:", "சுருக்கம்:", "Here is a summary:", "The summary is:"]
        
        for prefix in prefixes:
            if summary.startswith(prefix):
                summary = summary[len(prefix):].strip()
        
        # Remove extra whitespace
        summary = ' '.join(summary.split())
        
        # Ensure proper ending
        if language in ["ml", "hi", "ta"]:
            if not summary.endswith("।") and not summary.endswith("."):
                summary += "।" if language == "hi" else "."
        else:
            if not summary.endswith("."):
                summary += "."
        
        return summary
    
    def summarize_by_sentences(self, text: str, max_sentences: int = 3) -> str:
        """Fallback summarizer for when Ollama is not available"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) <= max_sentences:
            return '. '.join(sentences) + '.'
        
        summary = '. '.join(sentences[:max_sentences]) + '.'
        
        if len(summary) > 500:
            if '.' in summary[:497]:
                cutoff = summary[:497].rfind('.') + 1
                summary = summary[:cutoff]
            else:
                summary = summary[:497] + "..."
        
        return summary

# Create global summarizer instance
summarizer = OllamaSummarizer()

def summarize_text(text: str, language: str = "en", use_ollama: bool = True) -> str:
    """Main summarization function with Ollama fallback"""
    print(f"\n📊 SUMMARIZATION REQUEST")
    print(f"Input length: {len(text)} characters")
    print(f"Language: {language}")
    print(f"Use Ollama: {use_ollama}")
    
    # Input validation
    if not text or len(text.strip()) < 50:
        print("⚠️  Text too short, returning as-is")
        return text
    
    # Check if text is already an error message
    if text.startswith("[") and "]" in text:
        print("⚠️  Text appears to be an error message, returning as-is")
        return text
    
    if use_ollama:
        # Try Ollama first
        ollama_summary = summarizer.summarize_with_ollama(text, language)
        
        # Check if Ollama summary is valid (not an error message)
        if not ollama_summary.startswith("[Error:"):
            return ollama_summary
        else:
            print(f"⚠️  Ollama failed, using fallback: {ollama_summary[:50]}")
    
    # Fallback to sentence-based summarization
    print("🔄 Using fallback summarizer")
    return summarizer.summarize_by_sentences(text)

def check_ollama_status() -> dict:
    """Check Ollama service status and available models"""
    return {
        "ollama_available": summarizer.check_connection(),
        "model": summarizer.model,
        "base_url": summarizer.base_url,
        "timestamp": time.time()
    }

# Test function
def test_ollama():
    """Test Ollama connection and summarization"""
    print("🧪 Testing Ollama connection...")
    
    if summarizer.check_connection():
        test_text = "Artificial intelligence is transforming many industries. Machine learning algorithms can now recognize patterns in data that humans cannot see. This technology is being used in healthcare for disease diagnosis, in finance for fraud detection, and in transportation for autonomous vehicles. The future of AI looks promising with continued advancements in deep learning and neural networks."
        
        print("\n🧪 Testing summarization...")
        summary = summarizer.summarize_with_ollama(test_text)
        print(f"✅ Test summary: {summary}")
        
        return True
    else:
        print("❌ Ollama not available")
        return False

if __name__ == "__main__":
    # Run test when module is executed directly
    test_ollama()