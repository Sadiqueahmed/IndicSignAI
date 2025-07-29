from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

class TranslationModel:
    def __init__(self):
        self.model_name = "facebook/nllb-200-distilled-600M"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()
        
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self.translator = pipeline(
            "translation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            src_lang="eng_Latn",
            tgt_lang="asm_Beng"
        )
    
    def translate(self, text):
        try:
            result = self.translator(text)
            return result[0]["translation_text"]
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return None