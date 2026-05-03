# translation.py - Unified NLLB Translation Engine with Dzongkha
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import logging
import traceback
from typing import Optional
import os
import re

# ── deep-translator (GoogleTranslator) — resilient free-tier fallback ──
try:
    from deep_translator import GoogleTranslator
    _DEEP_TRANSLATOR_OK = True
except ImportError:
    _DEEP_TRANSLATOR_OK = False
    print('[WARN] deep-translator not installed. Run: pip install deep-translator')

# FALLBACK_DICTIONARY from english.py (best-effort; not required for core operation)
try:
    from src.english import FALLBACK_DICTIONARY          # package import (run.py / pytest)
except ImportError:
    try:
        from english import FALLBACK_DICTIONARY          # standalone: python translation.py
    except ImportError:
        FALLBACK_DICTIONARY = {}

# ── Meitei Mayek script enforcement ──
try:
    from src.models.bengali_to_meitei import ensure_meitei_mayek   # package import
except ImportError:
    try:
        from .bengali_to_meitei import ensure_meitei_mayek          # relative import
    except ImportError:
        try:
            from models.bengali_to_meitei import ensure_meitei_mayek  # legacy standalone
        except ImportError:
            try:
                from bengali_to_meitei import ensure_meitei_mayek     # cwd=src/ standalone
            except ImportError:
                # Last-resort no-op so the engine still boots
                def ensure_meitei_mayek(text):  # type: ignore
                    return text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "facebook/nllb-200-distilled-600M"
        
        # Language code mapping for NLLB-200
        # ─────────────────────────────────────────────────────────────────
        # Manipuri / Meitei Mayek:  'mni_Mtei'
        #   • mni = ISO 639-3 code for Meitei (Manipuri)
        #   • Mtei = ISO 15924 script tag for Meitei Mayek (U+ABC0–U+ABFF)
        #
        # WARNING: NLLB may fall back to 'mni_Beng' (Bengali script) if the
        # Meitei Mayek beam search fails or the source sentence is OOV.
        # ensure_meitei_mayek() is called on all output to catch this and
        # transliterate Bengali→Meitei Mayek char-by-char.
        # ─────────────────────────────────────────────────────────────────
        self.lang_codes = {
            'assamese': 'asm_Beng',
            'hindi': 'hin_Deva',
            'manipuri': 'mni_Mtei',  # ← Meitei Mayek script (NOT mni_Beng)
            'nepali': 'npi_Deva',
            'marathi': 'mar_Deva',
            'odia': 'ory_Orya',
            'mizorami': 'lus_Latn',
            'gujarati': 'guj_Gujr',
            'tamil': 'tam_Taml',
            'telugu': 'tel_Telu',
            'bengali': 'ben_Beng',
            'english': 'eng_Latn',
            'dzongkha': 'dzo_Tibt',  # Added Dzongkha
            'meitei_lon': 'custom'   # Special handling
        }
        
        # ── deep-translator (GoogleTranslator) language codes ──
        # Maps our internal language keys to ISO 639-1 codes used by Google.
        # None = not supported by Google Translate; skip deep-translator for that lang.
        self.deep_translator_codes = {
            'assamese': 'as',
            'hindi': 'hi',
            'manipuri': 'mni-Mtei',   # Google Translate supports Meitei Mayek script
            'nepali': 'ne',
            'marathi': 'mr',
            'odia': 'or',
            'mizorami': None,
            'gujarati': 'gu',
            'tamil': 'ta',
            'telugu': 'te',
            'bengali': 'bn',
            'english': 'en',
            'dzongkha': None,   # Not supported by Google Translate
            'meitei_lon': None,
        }
        
        # Initialize models and pipelines
        self.model = None
        self.tokenizer = None
        self.pipelines = {}
        self.fallback_translations = self._build_fallback_dictionary()
        
        # Try to load Meitei Lon model
        self.meitei_translator = None
        try:
            from src.models.meitei_lon_fallback import MeiteiLonFallback   # package import
            self.meitei_translator = MeiteiLonFallback()
            print("[OK] Meitei Lon fallback translator loaded")
        except ImportError:
            try:
                from meitei_lon_fallback import MeiteiLonFallback           # standalone
                self.meitei_translator = MeiteiLonFallback()
                print("[OK] Meitei Lon fallback translator loaded (standalone path)")
            except ImportError:
                print("! Meitei Lon fallback not available, using dictionary")
        
        # Load NLLB model
        self._load_nllb_model()
    
    def _load_nllb_model(self):
        """Load the NLLB model"""
        import gc
        try:
            print(f"Loading NLLB model on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
            print("[OK] NLLB model loaded successfully")
        except Exception as e:
            print(f"[X] Failed to load NLLB model: {e}")
            print("  Using fallback dictionary only")
            self.model = None
            self.tokenizer = None
            # Force garbage collection to prevent OS memory thrashing/swapping
            gc.collect()
    
    def _get_pipeline(self, target_lang):
        """Get or create translation pipeline for language"""
        if not self.model or not self.tokenizer:
            return None
            
        if target_lang not in self.pipelines:
            if target_lang not in self.lang_codes:
                raise ValueError(f"Unsupported language: {target_lang}")
            
            tgt_code = self.lang_codes[target_lang]
            if tgt_code == 'custom':
                return None
            
            print(f"Creating pipeline for {target_lang} ({tgt_code})...")
            
            self.pipelines[target_lang] = pipeline(
                "translation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                src_lang="eng_Latn",
                tgt_lang=tgt_code,
                max_length=200
            )
        
        return self.pipelines[target_lang]
    
    def _build_fallback_dictionary(self):
        """Enhanced fallback dictionary for all languages including Dzongkha"""
        return {
            # Greetings
            'hello': {
                'assamese': 'নমস্কাৰ', 
                'hindi': 'नमस्ते', 
                'manipuri': 'ꯑꯍꯥꯟꯕ',
                'nepali': 'नमस्ते', 
                'meitei_lon': 'ꯑꯍꯥꯟꯕ', 
                'marathi': 'नमस्कार',
                'tamil': 'வணக்கம்', 
                'bengali': 'হ্যালো',
                'dzongkha': 'ཡ་ཨ་ན',  # Dzongkha greeting
                'mizorami': 'Chibai',
                'odia': 'ନମସ୍କାର'
            },
            'hi': {
                'assamese': 'নমস্কাৰ', 
                'hindi': 'नमस्ते', 
                'manipuri': 'ꯍꯥꯏ',
                'nepali': 'नमस्ते', 
                'meitei_lon': 'ꯍꯥꯏ', 
                'tamil': 'வணக்கம்',
                'dzongkha': 'ཡ་ཨ་ན'
            },
            'namaste': {
                'assamese': 'নমস্কাৰ', 
                'hindi': 'नमस्ते', 
                'manipuri': 'ꯅꯃꯁ꯭ꯇꯦ',
                'nepali': 'नमस्ते',
                'meitei_lon': 'ꯅꯃꯁ꯭ꯇꯦ',
                'dzongkha': 'ཞུ་གསོལ'
            },
            'thank you': {
                'assamese': 'ধন্যবাদ', 
                'hindi': 'धन्यवाद', 
                'manipuri': 'ꯊꯥꯒꯠꯆꯔꯤ',
                'nepali': 'धन्यवाद', 
                'meitei_lon': 'ꯊꯥꯒꯠꯆꯔꯤ', 
                'marathi': 'धन्यवाद',
                'tamil': 'நன்றி', 
                'bengali': 'ধন্যবাদ',
                'dzongkha': 'བཀའ་དྲིན་ཆེ',  # Dzongkha thank you
                'mizorami': 'Lawmthu',
                'odia': 'ଧନ୍ୟବାଦ'
            },
            'thanks': {
                'assamese': 'ধন্যবাদ', 
                'hindi': 'शुक्रिया', 
                'manipuri': 'ꯊꯥꯒꯠꯆꯔꯤ',
                'nepali': 'धन्यवाद', 
                'meitei_lon': 'ꯊꯥꯒꯠꯆꯔꯤ',
                'dzongkha': 'བཀའ་དྲིན་ཆེ'
            },
            'good morning': {
                'assamese': 'শুভ ৰাতিপুৱা', 
                'hindi': 'शुभ प्रभात', 
                'manipuri': 'ꯑꯍꯥꯟꯕ ꯑꯌꯨꯛ', 
                'nepali': 'शुभ बिहान',
                'meitei_lon': 'ꯑꯍꯥꯟꯕ ꯑꯌꯨꯛ', 
                'tamil': 'காலை வணக்கம்',
                'dzongkha': 'ང་ག་དྲོ་པ་བདེ་ལེགས',  # Dzongkha good morning
                'mizorami': 'Chibai',
                'odia': 'ସୁପ୍ରଭାତ'
            },
            'good evening': {
                'assamese': 'শুভ সন্ধ্যা', 
                'hindi': 'शुभ संध्या',
                'manipuri': 'ꯑꯍꯥꯟꯕ ꯅꯨꯃꯤꯗꯥꯡꯊꯣꯏ', 
                'nepali': 'शुभ सन्ध्या',
                'meitei_lon': 'ꯑꯍꯥꯟꯕ ꯅꯨꯃꯤꯗꯥꯡꯊꯣꯏ',
                'dzongkha': 'ང་ག་དགོང་མོ་བདེ་ལེགས',  # Dzongkha good evening
                'tamil': 'மாலை வணக்கம்'
            },
            'how are you': {
                'assamese': 'আপোনাৰ কেনে আছে', 
                'hindi': 'आप कैसे हैं',
                'manipuri': 'ꯑꯗꯣꯝ ꯀꯝꯗꯧꯔꯤ', 
                'nepali': 'तपाईंलाई कस्तो छ',
                'meitei_lon': 'ꯑꯗꯣꯝ ꯀꯝꯗꯧꯔꯤ', 
                'tamil': 'நீங்கள் எப்படி இருக்கிறீர்கள்',
                'dzongkha': 'ག་དེ་འབད་ཡོད་',  # Dzongkha how are you
                'mizorami': 'I dam em?',
                'odia': 'କେମିତି ଅଛନ୍ତି'
            },
            'i am fine': {
                'assamese': 'মই ভাল আছোঁ', 
                'hindi': 'मैं ठीक हूँ',
                'manipuri': 'ꯑꯩ ꯐꯖꯅ ꯂꯩ', 
                'nepali': 'म सन्चै छु',
                'meitei_lon': 'ꯑꯩ ꯐꯖꯅ ꯂꯩ',
                'dzongkha': 'ང་བདེ་པོ་ཡོད་',  # Dzongkha I am fine
                'tamil': 'நான் நலமாக இருக்கிறேன்'
            },
            'what is your name': {
                'assamese': 'আপোনাৰ নাম কি', 
                'hindi': 'आपका नाम क्या है',
                'manipuri': 'ꯑꯗꯣꯝꯒꯤ ꯃꯤꯡ ꯀꯔꯤꯅꯣ', 
                'nepali': 'तपाईंको नाम के हो',
                'meitei_lon': 'ꯑꯗꯣꯝꯒꯤ ꯃꯤꯡ ꯀꯔꯤꯅꯣ', 
                'tamil': 'உங்கள் பெயர் என்ன',
                'dzongkha': 'ཁྱེད་རང་གི་མིང་ག་ཅི་ཨིན་ན་',  # Dzongkha what's your name
                'mizorami': 'I hming eng nge?'
            },
            'my name is': {
                'assamese': 'মোৰ নাম', 
                'hindi': 'मेरा नाम',
                'manipuri': 'ꯑꯩꯒꯤ ꯃꯤꯡ', 
                'nepali': 'मेरो नाम',
                'meitei_lon': 'ꯑꯩꯒꯤ ꯃꯤꯡ', 
                'tamil': 'என் பெயர்',
                'dzongkha': 'ངའི་མིང་',
                'odia': 'ମୋର ନାମ'
            },
            'water': {
                'assamese': 'পানী', 
                'hindi': 'पानी', 
                'manipuri': 'ꯏꯁꯤꯡ',
                'nepali': 'पानी', 
                'meitei_lon': 'ꯏꯁꯤꯡ', 
                'tamil': 'தண்ணீர்',
                'dzongkha': 'ཆུ',  # Dzongkha water
                'mizorami': 'Tui',
                'odia': 'ପାଣି'
            },
            'food': {
                'assamese': 'খাদ্য', 
                'hindi': 'भोजन', 
                'manipuri': 'ꯆꯥ',
                'nepali': 'खाना', 
                'meitei_lon': 'ꯆꯥ', 
                'tamil': 'உணவு',
                'dzongkha': 'ཟ་བ',  # Dzongkha food
                'mizorami': 'Ei',
                'odia': 'ଖାଦ୍ୟ'
            },
            'help': {
                'assamese': 'সহায়', 
                'hindi': 'मदद', 
                'manipuri': 'ꯃꯇꯦꯡ',
                'nepali': 'मद्दत', 
                'meitei_lon': 'ꯃꯇꯦꯡ', 
                'tamil': 'உதவி',
                'dzongkha': 'རོགས་པ',  # Dzongkha help
                'odia': 'ସହାୟତା'
            },
            'yes': {
                'assamese': 'হয়', 
                'hindi': 'हाँ', 
                'manipuri': 'ꯍꯣꯏ',
                'nepali': 'हो', 
                'meitei_lon': 'ꯍꯣꯏ', 
                'tamil': 'ஆம்',
                'dzongkha': 'ཨིན',  # Dzongkha yes
                'odia': 'ହଁ'
            },
            'no': {
                'assamese': 'নহয়', 
                'hindi': 'नहीं', 
                'manipuri': 'ꯅꯠꯇꯦ',
                'nepali': 'होइन', 
                'meitei_lon': 'ꯅꯠꯇꯦ', 
                'tamil': 'இல்லை',
                'dzongkha': 'མེན',  # Dzongkha no
                'odia': 'ନାଁ'
            },
            'please': {
                'assamese': 'অনুগ্ৰহ কৰি', 
                'hindi': 'कृपया', 
                'manipuri': 'ꯆꯥꯟꯕꯤꯗꯨꯅꯥ',
                'nepali': 'कृपया', 
                'meitei_lon': 'ꯆꯥꯟꯕꯤꯗꯨꯅꯥ', 
                'tamil': 'தயவு செய்து',
                'dzongkha': 'གུ་མ་ཞུ་ག',  # Dzongkha please
                'odia': 'ଦୟାକରି'
            },
            'sorry': {
                'assamese': 'দুঃখিত', 
                'hindi': 'क्षमा करें', 
                'manipuri': 'ꯁꯣꯔꯤ',
                'nepali': 'माफ गर्नुहोस्', 
                'meitei_lon': 'ꯁꯣꯔꯤ', 
                'tamil': 'மன்னிக்கவும்',
                'dzongkha': 'དགོངས་མ་ཚོམ',  # Dzongkha sorry
                'odia': 'ଦୁଃଖିତ'
            },
            'good': {
                'assamese': 'ভাল', 
                'hindi': 'अच्छा', 
                'manipuri': 'ꯐꯖꯔꯤ',
                'nepali': 'राम्रो', 
                'meitei_lon': 'ꯐꯖꯔꯤ', 
                'tamil': 'நல்ல',
                'dzongkha': 'བདེ་པོ',  # Dzongkha good
                'odia': 'ଭଲ'
            },
            'bad': {
                'assamese': 'বেয়া', 
                'hindi': 'बुरा', 
                'manipuri': 'ꯐꯖꯗꯦ',
                'nepali': 'खराब', 
                'meitei_lon': 'ꯐꯖꯗꯦ', 
                'tamil': 'கெட்ட',
                'dzongkha': 'མེད་པ',  # Dzongkha bad
                'odia': 'ଖରାପ'
            },
            'family': {
                'assamese': 'পৰিয়াল', 
                'hindi': 'परिवार', 
                'manipuri': 'ꯏꯃꯨꯡ',
                'nepali': 'परिवार', 
                'meitei_lon': 'ꯏꯃꯨꯡ', 
                'tamil': 'குடும்பம்',
                'dzongkha': 'ནང་མི་ཚང',  # Dzongkha family
                'odia': 'ପରିବାର'
            },
            'friend': {
                'assamese': 'বন্ধু', 
                'hindi': 'दोस्त', 
                'manipuri': 'ꯃꯔꯨꯞ',
                'nepali': 'साथी', 
                'meitei_lon': 'ꯃꯔꯨꯞ', 
                'tamil': 'நண்பர்',
                'dzongkha': 'རོགས་པ',  # Dzongkha friend
                'odia': 'ସାଙ୍ଗ'
            },
            'love': {
                'assamese': 'মৰম', 
                'hindi': 'प्यार', 
                'manipuri': 'ꯅꯨꯡꯁꯤ',
                'nepali': 'माया', 
                'meitei_lon': 'ꯅꯨꯡꯁꯤ', 
                'tamil': 'காதல்',
                'dzongkha': 'གཅེས་ཐགས',  # Dzongkha love
                'odia': 'ଭଲପାଇବା'
            },
            'india': {
                'assamese': 'ভাৰত', 
                'hindi': 'भारत', 
                'manipuri': 'ꯏꯟꯗꯤꯌꯥ',
                'nepali': 'भारत', 
                'meitei_lon': 'ꯏꯟꯗꯤꯌꯥ', 
                'tamil': 'இந்தியா',
                'dzongkha': 'རྒྱ་གར',  # Dzongkha India
                'odia': 'ଭାରତ'
            },
            'bhutan': {
                'assamese': 'ভূটান',
                'hindi': 'भूटान',
                'manipuri': 'ꯚꯨꯇꯥꯟ',
                'nepali': 'भुटान',
                'tamil': 'பூட்டான்',
                'dzongkha': 'འབྲུག་ཡུལ',  # Dzongkha Bhutan
                'english': 'Bhutan'
            },
            'thimphu': {
                'assamese': 'থিম্ফু',
                'hindi': 'थिम्फू',
                'dzongkha': 'ཐིམ་ཕུག',  # Dzongkha Thimphu
                'english': 'Thimphu'
            },
            'name': {
                'assamese': 'নাম', 
                'hindi': 'नाम', 
                'manipuri': 'ꯃꯤꯡ',
                'nepali': 'नाम', 
                'meitei_lon': 'ꯃꯤꯡ', 
                'tamil': 'பெயர்',
                'dzongkha': 'མིང་',  # Dzongkha name
                'odia': 'ନାମ'
            },
            'what': {
                'assamese': 'কি', 
                'hindi': 'क्या', 
                'manipuri': 'ꯀꯔꯤ',
                'nepali': 'के', 
                'meitei_lon': 'ꯀꯔꯤ', 
                'tamil': 'என்ன',
                'dzongkha': 'ག་ཅི་',  # Dzongkha what
                'odia': 'କଣ'
            },
            'where': {
                'assamese': 'ক', 
                'hindi': 'कहाँ', 
                'manipuri': 'ꯀꯗꯥꯏꯗ',
                'nepali': 'कहाँ', 
                'meitei_lon': 'ꯀꯗꯥꯏꯗ', 
                'tamil': 'எங்கே',
                'dzongkha': 'ག་པར་',  # Dzongkha where
                'odia': 'କୁଆଁ'
            },
            'when': {
                'assamese': 'কেতিয়া', 
                'hindi': 'कब', 
                'manipuri': 'ꯀꯔꯝꯕ ꯃꯇꯝꯗ',
                'nepali': 'कहिले', 
                'meitei_lon': 'ꯀꯔꯝꯕ ꯃꯇꯝꯗ', 
                'tamil': 'எப்போது',
                'dzongkha': 'ག་དུས་',  # Dzongkha when
                'odia': 'କେବେ'
            },
            'why': {
                'assamese': 'কিয়', 
                'hindi': 'क्यों', 
                'manipuri': 'ꯀꯔꯤꯒꯤ',
                'nepali': 'किन', 
                'meitei_lon': 'ꯀꯔꯤꯒꯤ', 
                'tamil': 'ஏன்',
                'dzongkha': 'ག་ཅིའི་ཆེད་',  # Dzongkha why
                'odia': 'କାହିଁକି'
            },
            'who': {
                'assamese': 'কোন', 
                'hindi': 'कौन', 
                'manipuri': 'ꯀꯅꯥ',
                'nepali': 'को', 
                'meitei_lon': 'ꯀꯅꯥ', 
                'tamil': 'யார்',
                'dzongkha': 'སུ་',  # Dzongkha who
                'odia': 'କିଏ'
            },
            'how': {
                'assamese': 'কিদৰে', 
                'hindi': 'कैसे', 
                'manipuri': 'ꯀꯔꯝꯅ',
                'nepali': 'कसरी', 
                'meitei_lon': 'ꯀꯔꯝꯅ', 
                'tamil': 'எப்படி',
                'dzongkha': 'ག་དེ་འབད་',  # Dzongkha how
                'odia': 'କେମିତି'
            },
            'day': {
                'assamese': 'দিন', 
                'hindi': 'दिन', 
                'manipuri': 'ꯅꯨꯃꯤꯠ',
                'nepali': 'दिन', 
                'meitei_lon': 'ꯅꯨꯃꯤꯠ', 
                'tamil': 'நாள்',
                'dzongkha': 'ཉིནམ་',  # Dzongkha day
                'odia': 'ଦିନ'
            },
            'night': {
                'assamese': 'ৰাতি', 
                'hindi': 'रात', 
                'manipuri': 'ꯑꯋꯥ',
                'nepali': 'रात', 
                'meitei_lon': 'ꯑꯋꯥ', 
                'tamil': 'இரவு',
                'dzongkha': 'མཚན་མོ་',  # Dzongkha night
                'odia': 'ରାତି'
            },
            'morning': {
                'assamese': 'পুৱা', 
                'hindi': 'सुबह', 
                'manipuri': 'ꯑꯌꯨꯛ',
                'nepali': 'बिहान', 
                'meitei_lon': 'ꯑꯌꯨꯛ', 
                'tamil': 'காலை',
                'dzongkha': 'དྲོ་པ་',  # Dzongkha morning
                'odia': 'ସକାଳ'
            },
            'book': {
                'assamese': 'কিতাপ', 
                'hindi': 'किताब', 
                'manipuri': 'ꯂꯥꯏꯔꯤꯛ',
                'nepali': 'किताब', 
                'meitei_lon': 'ꯂꯥꯏꯔꯤꯛ', 
                'tamil': 'புத்தகம்',
                'dzongkha': 'དཔེ་ཆ་',  # Dzongkha book
                'odia': 'ବହି'
            },
            'school': {
                'assamese': 'বিদ্যালয়', 
                'hindi': 'स्कूल', 
                'manipuri': 'ꯃꯇꯝꯂꯣꯟ',
                'nepali': 'स्कूल', 
                'meitei_lon': 'ꯃꯇꯝꯂꯣꯟ', 
                'tamil': 'பள்ளி',
                'dzongkha': 'སློབ་གྲྭ',  # Dzongkha school
                'odia': 'ସ୍କୁଲ'
            },
            'teacher': {
                'assamese': 'শিক্ষক', 
                'hindi': 'शिक्षक', 
                'manipuri': 'ꯑꯣꯖꯥ',
                'nepali': 'शिक्षक', 
                'meitei_lon': 'ꯑꯣꯖꯥ', 
                'tamil': 'ஆசிரியர்',
                'dzongkha': 'དགེ་རྒན',  # Dzongkha teacher
                'odia': 'ଶିକ୍ଷକ'
            },
            'student': {
                'assamese': 'ছাত্ৰ', 
                'hindi': 'छात्र', 
                'manipuri': 'ꯃꯃꯤꯡ',
                'nepali': 'विद्यार्थी', 
                'meitei_lon': 'ꯃꯃꯤꯡ', 
                'tamil': 'மாணவர்',
                'dzongkha': 'སློབ་མ',  # Dzongkha student
                'odia': 'ଛାତ୍ର'
            },
            'house': {
                'assamese': 'ঘৰ', 
                'hindi': 'घर', 
                'manipuri': 'ꯌꯨꯝ',
                'nepali': 'घर', 
                'meitei_lon': 'ꯌꯨꯝ', 
                'tamil': 'வீடு',
                'dzongkha': 'ནང་ཁྱིམ',  # Dzongkha house
                'odia': 'ଘର'
            }
        }
    
    def translate(self, text: str, target_lang: str = "hindi") -> Optional[str]:
        """
        Translate English text to target language using NLLB model
        Falls back to dictionary if NLLB fails.
        
        For Manipuri: ALWAYS enforces Meitei Mayek script output,
        converting Bengali script if NLLB returns it.
        """
        if not text or not text.strip():
            return text
        
        # Special handling for Meitei Lon
        if target_lang == "meitei_lon":
            return self._translate_meitei(text)
        
        # ── PRIMARY: deep-translator (GoogleTranslator) ──────────────
        # This is the resilient primary layer. Unlike NLLB it handles full
        # sentences gracefully without partial translations, and covers hi/as/ne/mr/or/ta/te/bn/gu.
        gt_code = self.deep_translator_codes.get(target_lang)
        if _DEEP_TRANSLATOR_OK and gt_code is not None:
            try:
                dt_result = GoogleTranslator(source='en', target=gt_code).translate(text)
            except Exception as e:
                logger.error(
                    f'[deep-translator] en->{target_lang} failed for '
                    f'input="{text}": {type(e).__name__}: {e}'
                )
                dt_result = None
            else:
                if dt_result and dt_result.strip():
                    logger.info(f'[OK] deep-translator {target_lang}: "{text}" -> "{dt_result}"')
                    if target_lang == 'manipuri':
                        dt_result = ensure_meitei_mayek(dt_result)
                    return dt_result

        # ── SECONDARY: NLLB Model (for Dzongkha, Meitei, or if deep-translator fails) ──
        if self.model and self.tokenizer:
            try:
                pipe = self._get_pipeline(target_lang)
                if pipe:
                    result = pipe(text)
                    if result and len(result) > 0:
                        translated = result[0]['translation_text']
                        logger.info(f'[OK] NLLB {target_lang}: "{text}" -> "{translated}"')
                        # ── MEITEI MAYEK ENFORCEMENT ────────────────────────────
                        if target_lang == 'manipuri':
                            before = translated
                            translated = ensure_meitei_mayek(translated)
                            if before != translated:
                                logger.info(
                                    f'[Meitei] NLLB returned Bengali; '
                                    f'transliterated: "{before}" -> "{translated}"'
                                )
                            if translated.lower().strip() == text.lower().strip():
                                logger.warning(
                                    f'[Meitei] NLLB echoed source text for input="{text}"; '
                                    f'falling through to dictionary.'
                                )
                            elif translated and translated.strip():
                                return translated
                        else:
                            return translated
            except Exception as e:
                logger.error(
                    f'[NLLB] translation failed for lang="{target_lang}" '
                    f'input="{text}": {type(e).__name__}: {e}'
                )

        # ── FALLBACK: dictionary ──────────────────────────────────────
        dict_result = self._dictionary_translate(text, target_lang)
        # For Manipuri: if dictionary returned empty (no matches), provide a
        # Meitei Mayek marker so the UI always shows something in the correct script.
        if target_lang == 'manipuri' and not dict_result:
            return f'ꯏꯟꯗꯤꯌꯥ ꯁꯥꯏꯜ ꯂꯦꯡꯉꯨ: {text}'  # "ISL sign:" in Meitei Mayek
        return dict_result
    
    def translate_with_nllb_direct(self, text: str, target_lang: str = "dzongkha") -> Optional[str]:
        """
        Direct translation using model10.py approach (for Dzongkha)
        """
        if not text or not text.strip():
            return text
        
        try:
            # Get language code
            tgt_code = self.lang_codes.get(target_lang, "dzo_Tibt")
            
            # Set source language
            self.tokenizer.src_lang = "eng_Latn"
            
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            # Generate with parameters similar to model10.py
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_code),
                max_length=100,
                num_beams=5,
                early_stopping=True
            )
            
            # Decode
            translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"[OK] Direct NLLB {target_lang}: '{text}' -> '{translated}'")
            return translated
            
        except Exception as e:
            logger.error(f"Direct NLLB translation failed: {e}")
            return self._dictionary_translate(text, target_lang)
    
    def _translate_meitei(self, text: str) -> str:
        """Translate to Meitei Lon using specialized model"""
        if self.meitei_translator:
            try:
                return self.meitei_translator.translate_to_meitei_lon(text)
            except Exception as e:
                logger.error(f"Meitei model failed: {e}")
        
        # Fallback to dictionary
        return self._dictionary_translate(text, "meitei_lon")
    
    def _dictionary_translate(self, text: str, target_lang: str) -> str:
        """Dictionary-based fallback translation"""
        text_lower = text.lower().strip()
        
        # Check for exact phrase matches
        if text_lower in self.fallback_translations:
            if target_lang in self.fallback_translations[text_lower]:
                result = self.fallback_translations[text_lower][target_lang]
                if target_lang == 'manipuri':
                    result = ensure_meitei_mayek(result)
                return result
        
        # Try common variations
        variations = {
            'hello': ['hi', 'hey', 'greetings', 'namaste'],
            'thank you': ['thanks', 'thankyou', 'thx'],
            'good morning': ['morning', 'gm'],
            'how are you': ['how r u', 'how are u', 'how do you do']
        }
        
        for phrase, var_list in variations.items():
            if text_lower in var_list and phrase in self.fallback_translations:
                if target_lang in self.fallback_translations[phrase]:
                    result = self.fallback_translations[phrase][target_lang]
                    if target_lang == 'manipuri':
                        result = ensure_meitei_mayek(result)
                    return result
        
        # Word-by-word translation
        words = text_lower.split()
        translated_words = []
        translated_count = 0
        
        for word in words:
            clean_word = ''.join(c for c in word if c.isalnum())
            
            if clean_word in self.fallback_translations and target_lang in self.fallback_translations[clean_word]:
                translated_words.append(self.fallback_translations[clean_word][target_lang])
                translated_count += 1
            else:
                translated_words.append(word)
        
        result = " ".join(translated_words)
        
        if translated_count > 0:
            # ── MEITEI MAYEK ENFORCEMENT on word-by-word result ──
            if target_lang == 'manipuri':
                result = ensure_meitei_mayek(result)
            return result
        else:
            # No dictionary matches at all.
            # For Manipuri, do NOT return a Latin-script placeholder — it will
            # bypass ensure_meitei_mayek() and corrupt the output.
            # Return empty string so the caller knows to treat this as a miss.
            if target_lang == 'manipuri':
                return ''
            return f"[{target_lang.upper()}] {text}"
    
    def translate_regional_to_english(self, text: str, source_lang: str) -> Optional[str]:
        """Translate regional language text to English.
        
        Priority: deep-translator (GoogleTranslator) → dictionary fallback.
        deep-translator is preferred because it handles full sentences;
        the dictionary only covers ~40 common words.
        """
        if not text or not text.strip():
            return text
        
        # ── PRIMARY: deep-translator (GoogleTranslator) ─────────────────
        gt_code = self.deep_translator_codes.get(source_lang)
        if _DEEP_TRANSLATOR_OK and gt_code is not None:
            try:
                dt_result = GoogleTranslator(source=gt_code, target='en').translate(text)
                if dt_result and dt_result.strip():
                    print(f"[OK] deep-translator {source_lang}->en: '{text}' -> '{dt_result}'")
                    return dt_result
            except Exception as e:
                print(f"[deep-translator] reverse FAILED for {source_lang}: {type(e).__name__}: {e}")
                print(traceback.format_exc())
                logger.error(
                    f'[deep-translator] {source_lang}->en failed for '
                    f'input="{text}": {type(e).__name__}: {e}'
                )
        
        # ── FALLBACK: dictionary reverse lookup ─────────────────────────
        reverse_map = {}
        for english_word, translations in self.fallback_translations.items():
            for lang, regional_word in translations.items():
                if lang not in reverse_map:
                    reverse_map[lang] = {}
                reverse_map[lang][regional_word.lower()] = english_word
        
        if source_lang in reverse_map:
            text_lower = text.lower().strip()
            if text_lower in reverse_map[source_lang]:
                return reverse_map[source_lang][text_lower]
            
            # Word by word
            words = text.split()
            translated_words = []
            for word in words:
                word_lower = word.lower()
                if word_lower in reverse_map[source_lang]:
                    translated_words.append(reverse_map[source_lang][word_lower])
                else:
                    translated_words.append(f"[{word}]")
            
            if len([w for w in translated_words if not w.startswith('[')]) > 0:
                return " ".join(translated_words)
        
        return f"[{source_lang.upper()}] {text}"

# Create global instance
translation_engine = TranslationModel()

def test_translations():
    """Test the translation system with all languages including Dzongkha"""
    print("\n" + "="*70)
    print("🧪 TESTING NLLB TRANSLATION ENGINE WITH DZONGKHA")
    print("="*70)
    
    test_cases = [
        ("Hello, how are you today?", "hindi"),
        ("Thank you for your help", "assamese"),
        ("I love my family", "manipuri"),
        ("What is your name?", "nepali"),
        ("Good morning", "tamil"),
        ("Can you help me please?", "bengali"),
        ("Where is the school?", "marathi"),
        ("Hello, my name is John", "dzongkha"),  # Test Dzongkha
        ("Thank you very much", "dzongkha"),     # Test Dzongkha
        ("How are you doing?", "dzongkha"),      # Test Dzongkha
        ("I am from India", "dzongkha"),         # Test Dzongkha
        ("What is this?", "odia"),
    ]
    
    for text, lang in test_cases:
        print(f"\n📝 English: {text}")
        
        # Use direct method for Dzongkha, regular for others
        if lang == "dzongkha" and translation_engine.model:
            result = translation_engine.translate_with_nllb_direct(text, lang)
        else:
            result = translation_engine.translate(text, lang)
        
        print(f"➡️  {lang.capitalize()}: {result}")
    
    print("\n" + "="*70)
    print("✅ Translation tests complete")
    print("="*70)

if __name__ == "__main__":
    test_translations()