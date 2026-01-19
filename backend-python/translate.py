from deep_translator import GoogleTranslator
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Language code mapping
LANGUAGE_MAP = {
    'ml': {'code': 'ml', 'name': 'Malayalam'},
    'hi': {'code': 'hi', 'name': 'Hindi'},
    'en': {'code': 'en', 'name': 'English'},
    'es': {'code': 'es', 'name': 'Spanish'},
    'fr': {'code': 'fr', 'name': 'French'},
    'ta': {'code': 'ta', 'name': 'Tamil'},
    'te': {'code': 'te', 'name': 'Telugu'},
    'kn': {'code': 'kn', 'name': 'Kannada'},
    'de': {'code': 'de', 'name': 'German'},
    'zh': {'code': 'zh-cn', 'name': 'Chinese'},
    'ja': {'code': 'ja', 'name': 'Japanese'},
    'ko': {'code': 'ko', 'name': 'Korean'},
    'ar': {'code': 'ar', 'name': 'Arabic'},
    'ru': {'code': 'ru', 'name': 'Russian'}
}

def translate_text(text, target_lang="en"):
    """Translate text using Google Translate with robust error handling"""
    print(f"\n🌐 TRANSLATION REQUEST")
    print(f"Target: {LANGUAGE_MAP.get(target_lang, {}).get('name', target_lang)}")
    print(f"Text length: {len(text)} characters")
    
    # Input validation
    if not text or not isinstance(text, str):
        print("❌ Invalid input text")
        return "[Error: No text to translate]"
    
    text = text.strip()
    if len(text) == 0:
        print("⚠️  Empty text")
        return ""
    
    # Check if text is already an error message
    if text.startswith("[") and "]" in text:
        print("⚠️  Text appears to be an error message, returning as-is")
        return text
    
    try:
        # Get language code
        lang_info = LANGUAGE_MAP.get(target_lang, LANGUAGE_MAP['en'])
        lang_code = lang_info['code']
        lang_name = lang_info['name']
        
        print(f"🌍 Using language code: {lang_code} ({lang_name})")
        
        # Limit text length for API (Google Translate has limits)
        original_length = len(text)
        if original_length > 4500:
            print(f"⚠️  Truncating text from {original_length} to 4500 characters")
            text = text[:4500] + " [...]"
        
        # Start translation
        print("⏳ Translating...")
        start_time = time.time()
        
        translator = GoogleTranslator(source='auto', target=lang_code)
        translated = translator.translate(text)
        
        translation_time = time.time() - start_time
        
        print(f"✅ Translation complete ({translation_time:.1f}s)")
        print(f"📊 Original: {original_length} chars")
        print(f"📊 Translated: {len(translated)} chars")
        
        if translated:
            preview = translated[:100] + "..." if len(translated) > 100 else translated
            print(f"📄 Preview: {preview}")
        else:
            print("⚠️  Empty translation returned")
            translated = "[Translation returned empty]"
        
        return translated
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ TRANSLATION FAILED")
        print(f"Error: {error_msg}")
        
        # Common error patterns
        if "quota" in error_msg.lower() or "limit" in error_msg.lower():
            logger.error("Google Translate API quota may be exceeded")
            return f"[Translation error: API limit reached. Please try again later.]\n\nOriginal text: {text[:200]}..."
        elif "network" in error_msg.lower() or "connection" in error_msg.lower():
            logger.error("Network error during translation")
            return f"[Translation error: Network issue. Check internet connection.]\n\nOriginal text: {text[:200]}..."
        elif "language" in error_msg.lower() or "supported" in error_msg.lower():
            logger.error(f"Unsupported language: {target_lang}")
            return f"[Translation error: Language '{target_lang}' not supported.]\n\nOriginal text: {text[:200]}..."
        else:
            logger.error(f"Translation error: {error_msg}")
            return f"[Translation error: {error_msg[:80]}]\n\nOriginal text: {text[:200]}..."

def get_supported_languages():
    """Get list of supported languages"""
    return list(LANGUAGE_MAP.keys())

def get_language_name(lang_code):
    """Get language name from code"""
    return LANGUAGE_MAP.get(lang_code, {}).get('name', 'Unknown')