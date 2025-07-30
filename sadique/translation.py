from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from typing import Optional

class TranslationModel:
    def __init__(self):
        self.model_name = "facebook/nllb-200-distilled-600M"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.supported_languages = {
            "assamese": "asm_Beng",
            "hindi": "hin_Deva",
            "manipuri_bangoli": "mni_Beng",
            "nepali": "npi_Deva"  # Verified correct code for Nepali
        }
        self.pipelines = {lang: None for lang in self.supported_languages}
        self.load_models()
        
    def load_models(self):
        """Load the shared model and create all translation pipelines"""
        try:
            print("Loading translation model...")
            
            # Load shared tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
            
            # Verify language support
            available_codes = set(self.tokenizer.lang_code_to_id.keys())
            print(f"Available language codes: {available_codes}")
            
            # Create all translation pipelines
            for lang_name, lang_code in self.supported_languages.items():
                if lang_code in available_codes:
                    self.pipelines[lang_name] = self._create_pipeline(lang_code)
                    print(f"Created pipeline for {lang_name} ({lang_code})")
                else:
                    print(f"Warning: Language code {lang_code} not supported for {lang_name}")
            
            print("Translation model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise
    
    def _create_pipeline(self, tgt_lang: str):
        """Helper to create translation pipeline with error handling"""
        try:
            return pipeline(
                "translation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                src_lang="eng_Latn",
                tgt_lang=tgt_lang,
                max_length=400  # Prevents truncation
            )
        except Exception as e:
            print(f"Error creating pipeline for {tgt_lang}: {str(e)}")
            return None
    
    def translate(self, text: str, target_lang: str = "assamese") -> Optional[str]:
        """
        Translate text to target language with robust error handling
        Returns None if translation fails
        """
        if not text or not text.strip():
            return None
            
        if target_lang not in self.pipelines:
            print(f"Unsupported language requested: {target_lang}")
            return None
            
        pipeline = self.pipelines[target_lang]
        if not pipeline:
            print(f"No pipeline available for {target_lang}")
            return None
            
        try:
            # Preprocess text to handle common issues
            text = text.strip()
            if len(text) > 1000:
                print("Warning: Truncating long text for translation")
                text = text[:1000]
                
            result = pipeline(text)
            
            if not result or not isinstance(result, list) or not result[0].get("translation_text"):
                print(f"Unexpected translation result format: {result}")
                return None
                
            translated = result[0]["translation_text"]
            
            # Verify translation actually changed
            if translated.lower() == text.lower():
                print(f"Warning: Translation returned original text for {target_lang}")
                return None
                
            return translated
            
        except Exception as e:
            print(f"Translation error ({target_lang}): {str(e)}")
            return None


# Test function to verify all languages
def test_translations():
    test_text = "Hello, how are you today?"
    translator = TranslationModel()
    
    for lang in translator.supported_languages:
        translation = translator.translate(test_text, lang)
        print(f"{lang.upper()}: {translation if translation else 'FAILED'}")


if __name__ == "__main__":
    # Run tests when executed directly
    print("Running translation tests...")
    test_translations()