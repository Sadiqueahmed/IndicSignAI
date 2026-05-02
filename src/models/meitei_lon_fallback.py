import logging
import re

logger = logging.getLogger(__name__)

class MeiteiLonFallback:
    """Robust fallback Meitei Lon translator"""
    
    def __init__(self):
        self.is_loaded = True
        self.fallback_mode = True
        print("[OK] Meitei Lon fallback translator initialized")
    
    def translate_to_meitei_lon(self, text: str) -> str:
        """English to Meitei Lon translation with comprehensive fallback"""
        if not text or not text.strip():
            return ""
        
        text_lower = text.lower().strip()
        
        # First, handle common contractions and phrases
        contractions = {
            "i'm": "i am",
            "you're": "you are",
            "he's": "he is",
            "she's": "she is",
            "it's": "it is",
            "we're": "we are",
            "they're": "they are",
            "i've": "i have",
            "you've": "you have",
            "we've": "we have",
            "they've": "they have",
            "i'll": "i will",
            "you'll": "you will",
            "he'll": "he will",
            "she'll": "she will",
            "it'll": "it will",
            "we'll": "we will",
            "they'll": "they will",
            "i'd": "i would",
            "you'd": "you would",
            "he'd": "he would",
            "she'd": "she would",
            "it'd": "it would",
            "we'd": "we would",
            "they'd": "they would",
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "can't": "cannot",
            "couldn't": "could not",
            "won't": "will not",
            "wouldn't": "would not",
            "shouldn't": "should not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "what's": "what is",
            "who's": "who is",
            "where's": "where is",
            "when's": "when is",
            "why's": "why is",
            "how's": "how is",
            "that's": "that is",
            "there's": "there is",
            "here's": "here is",
            "let's": "let us"
        }
        
        # Expand contractions first
        for contraction, expanded in contractions.items():
            text_lower = re.sub(r'\b' + contraction + r'\b', expanded, text_lower)
        
        # Comprehensive translation dictionary
        translations = {
            # Pronouns
            "i": "ꯑꯩ",
            "me": "ꯑꯩ",
            "my": "ꯑꯩꯒꯤ",
            "mine": "ꯑꯩꯒꯤ",
            "you": "ꯑꯗꯣꯝ",
            "your": "ꯑꯗꯣꯝꯒꯤ",
            "yours": "ꯑꯗꯣꯝꯒꯤ",
            "he": "ꯃꯍꯥꯛ",
            "him": "ꯃꯍꯥꯛ",
            "his": "ꯃꯍꯥꯛꯒꯤ",
            "she": "ꯃꯍꯥꯛ",
            "her": "ꯃꯍꯥꯛ",
            "hers": "ꯃꯍꯥꯛꯒꯤ",
            "it": "ꯃꯁꯤ",
            "its": "ꯃꯁꯤꯒꯤ",
            "we": "ꯑꯩꯈꯣꯏ",
            "us": "ꯑꯩꯈꯣꯏ",
            "our": "ꯑꯩꯈꯣꯏꯒꯤ",
            "ours": "ꯑꯩꯈꯣꯏꯒꯤ",
            "they": "ꯃꯈꯣꯏ",
            "them": "ꯃꯈꯣꯏ",
            "their": "ꯃꯈꯣꯏꯒꯤ",
            "theirs": "ꯃꯈꯣꯏꯒꯤ",
            
            # Common verbs - present tense
            "am": "ꯑꯣꯏ",
            "is": "ꯑꯣꯏ",
            "are": "ꯑꯣꯏ",
            "have": "ꯂꯩ",
            "has": "ꯂꯩ",
            "do": "ꯇꯧ",
            "does": "ꯇꯧ",
            "go": "ꯆꯠ",
            "goes": "ꯆꯠ",
            "come": "ꯂꯥ",
            "comes": "ꯂꯥ",
            "see": "ꯎꯕ",
            "sees": "ꯎꯕ",
            "look": "ꯎꯕ",
            "looks": "ꯎꯕ",
            "speak": "ꯉꯥꯡꯕ",
            "speaks": "ꯉꯥꯡꯕ",
            "talk": "ꯉꯥꯡꯕ",
            "talks": "ꯉꯥꯡꯕ",
            "eat": "ꯆꯥ",
            "eats": "ꯆꯥ",
            "drink": "ꯌꯦꯟ",
            "drinks": "ꯌꯦꯟ",
            "sleep": "ꯌꯦꯜ",
            "sleeps": "ꯌꯦꯜ",
            "walk": "ꯅꯀꯁꯤꯜ",
            "walks": "ꯅꯀꯁꯤꯜ",
            "run": "ꯅꯀꯁꯤꯜ ꯍꯦꯟꯅ",
            "runs": "ꯅꯀꯁꯤꯜ ꯍꯦꯟꯅ",
            "work": "ꯁꯤꯖꯤꯟꯅꯕ",
            "works": "ꯁꯤꯖꯤꯟꯅꯕ",
            "learn": "ꯇꯞꯅꯕ",
            "learns": "ꯇꯞꯅꯕ",
            "teach": "ꯇꯞꯆꯕ",
            "teaches": "ꯇꯞꯆꯕ",
            "know": "ꯈꯪ",
            "knows": "ꯈꯪ",
            "understand": "ꯈꯪꯅꯕ",
            "understands": "ꯈꯪꯅꯕ",
            "love": "ꯅꯨꯡꯁꯤ",
            "loves": "ꯅꯨꯡꯁꯤ",
            "like": "ꯅꯨꯡꯁꯤ",
            "likes": "ꯅꯨꯡꯁꯤ",
            "want": "ꯌꯦꯡꯁꯤꯟ",
            "wants": "ꯌꯦꯡꯁꯤꯟ",
            "need": "ꯃꯔꯝꯗ",
            "needs": "ꯃꯔꯝꯗ",
            "feel": "ꯉꯥꯡꯕ",
            "feels": "ꯉꯥꯡꯕ",
            
            # Common verbs - past tense
            "was": "ꯑꯣꯏꯈ꯭ꯔꯦ",
            "were": "ꯑꯣꯏꯈ꯭ꯔꯦ",
            "had": "ꯂꯩꯈ꯭ꯔꯦ",
            "did": "ꯇꯧꯈ꯭ꯔꯦ",
            "went": "ꯆꯠꯈ꯭ꯔꯦ",
            "came": "ꯂꯥꯈ꯭ꯔꯦ",
            "saw": "ꯎꯈ꯭ꯔꯦ",
            "ate": "ꯆꯥꯈ꯭ꯔꯦ",
            "drank": "ꯌꯦꯟꯈ꯭ꯔꯦ",
            "slept": "ꯌꯦꯜꯈ꯭ꯔꯦ",
            "walked": "ꯅꯀꯁꯤꯜꯈ꯭ꯔꯦ",
            "ran": "ꯅꯀꯁꯤꯜꯈ꯭ꯔꯦ",
            "worked": "ꯁꯤꯖꯤꯟꯅꯈ꯭ꯔꯦ",
            "learned": "ꯇꯞꯅꯈ꯭ꯔꯦ",
            "taught": "ꯇꯞꯆꯈ꯭ꯔꯦ",
            "knew": "ꯈꯪꯈ꯭ꯔꯦ",
            "understood": "ꯈꯪꯅꯈ꯭ꯔꯦ",
            "loved": "ꯅꯨꯡꯁꯤꯈ꯭ꯔꯦ",
            "liked": "ꯅꯨꯡꯁꯤꯈ꯭ꯔꯦ",
            "wanted": "ꯌꯦꯡꯁꯤꯟꯈ꯭ꯔꯦ",
            "needed": "ꯃꯔꯝꯗꯈ꯭ꯔꯦ",
            "felt": "ꯉꯥꯡꯈ꯭ꯔꯦ",
            
            # Common adjectives
            "very": "ꯌꯥꯝꯅꯅ",
            "so": "ꯌꯥꯝꯅꯅ",
            "too": "ꯌꯥꯝꯅꯅ",
            "much": "ꯌꯥꯝꯅꯅ",
            "many": "ꯌꯥꯝꯅꯅ",
            "more": "ꯍꯦꯟꯅ",
            "less": "ꯈ꯭ꯋꯥꯏꯗ",
            "good": "ꯐꯖꯔꯤ",
            "bad": "ꯐꯖꯗꯦ",
            "happy": "ꯅꯨꯡꯉꯥꯏ",
            "sad": "ꯅꯨꯡꯁꯥ",
            "angry": "ꯑꯅꯥꯎ",
            "beautiful": "ꯅꯨꯡꯁꯥꯗ꯭ꯔꯕ",
            "ugly": "ꯅꯨꯡꯁꯥꯗ꯭ꯔꯕꯗꯦ",
            "big": "ꯑꯍꯥꯟꯕ",
            "small": "ꯈ꯭ꯋꯥꯏ",
            "new": "ꯑꯅꯧꯕ",
            "old": "ꯍꯩꯊꯣꯏꯕ",
            "young": "ꯅꯍꯥ",
            "hot": "ꯑꯉꯥꯡ",
            "cold": "ꯑꯇꯣꯞ",
            "rich": "ꯌꯥꯝꯕ",
            "poor": "ꯌꯥꯝꯕꯗꯦ",
            "strong": "ꯁꯤꯡꯕ",
            "weak": "ꯁꯤꯡꯕꯗꯦ",
            "fast": "ꯈꯨꯗꯝ",
            "slow": "ꯈꯨꯗꯝꯗꯦ",
            
            # Common adverbs
            "now": "ꯍꯧꯖꯤꯛ",
            "today": "ꯉꯁꯤ",
            "tomorrow": "ꯍꯌꯦꯡ",
            "yesterday": "ꯑꯌꯦꯡ",
            "always": "ꯄꯨꯃꯅꯨꯡ",
            "never": "ꯅꯠꯇꯦ",
            "sometimes": "ꯀꯌꯥꯗ",
            "often": "ꯌꯥꯝꯅꯅ",
            "soon": "ꯃꯇꯨꯡꯗ",
            "later": "ꯃꯇꯨꯡꯗ",
            "here": "ꯃꯐꯝꯗ",
            "there": "ꯃꯐꯝꯗ",
            "everywhere": "ꯄꯨꯔꯛꯐꯝꯗ",
            
            # Common nouns
            "name": "ꯃꯤꯡ",
            "house": "ꯌꯨꯝ",
            "home": "ꯎꯃꯪ",
            "food": "ꯆꯥ",
            "water": "ꯏꯁꯤꯡ",
            "book": "ꯄꯨꯊꯣꯛ",
            "school": "ꯁ꯭ꯀꯨꯜ",
            "city": "ꯇꯦꯡꯀꯣꯏꯕ",
            "country": "ꯂꯩꯄꯥꯛ",
            "world": "ꯂꯩꯄꯥꯛ",
            "person": "ꯃꯤꯑꯣꯏ",
            "people": "ꯃꯤꯑꯣꯏꯁꯤꯡ",
            "man": "ꯅꯨꯄꯥ",
            "woman": "ꯅꯨꯄꯤ",
            "child": "ꯃꯔꯨꯞ",
            "friend": "ꯏꯔꯩꯁꯥ",
            "family": "ꯏꯃꯥ",
            "mother": "ꯅꯣꯡꯃ",
            "father": "ꯄꯣꯛꯄ",
            "brother": "ꯏꯄꯥ",
            "sister": "ꯏꯄꯨ",
            "son": "ꯁꯥ",
            "daughter": "ꯁꯥꯔꯨ",
            
            # Places
            "manipur": "ꯃꯅꯤꯄꯨꯔ",
            "imphal": "ꯏꯝꯐꯥꯜ",
            "india": "ꯏꯟꯗꯤꯌꯥ",
            
            # Greetings and common phrases
            "hello": "ꯑꯍꯥꯟꯕ",
            "hi": "ꯑꯍꯥꯟꯕ",
            "namaste": "ꯅꯃꯁ꯭ꯇꯦ",
            "good morning": "ꯑꯍꯥꯟꯕ ꯑꯌꯨꯛ",
            "good afternoon": "ꯑꯍꯥꯟꯕ ꯅꯨꯃꯤꯗꯥꯡꯊꯣꯏ",
            "good evening": "ꯑꯍꯥꯟꯕ ꯅꯨꯃꯤꯗꯥꯡꯊꯣꯏ",
            "good night": "ꯑꯍꯥꯟꯕ ꯑꯔꯤꯕ",
            "how are you": "ꯑꯗꯣꯝ ꯀꯝꯗꯧꯔꯤ",
            "what is your name": "ꯑꯗꯣꯝꯒꯤ ꯃꯤꯡ ꯀꯔꯤꯅꯣ",
            "my name is": "ꯑꯩꯒꯤ ꯃꯤꯡ",
            "thank you": "ꯊꯥꯒꯠꯆꯔꯤ",
            "thanks": "ꯊꯥꯒꯠꯆꯔꯤ",
            "please": "ꯆꯥꯟꯅꯅ",
            "sorry": "ꯊꯥꯒꯠꯆꯔꯤ",
            "excuse me": "ꯀꯔꯤꯒꯨꯝꯕ",
            "welcome": "ꯇꯔꯥꯟꯅ",
            
            # Basic questions
            "what": "ꯀꯔꯤ",
            "who": "ꯀꯅꯥ",
            "where": "ꯀꯗꯥꯏꯗ",
            "when": "ꯀꯌꯥꯗ",
            "why": "ꯀꯔꯤꯒꯤꯗꯃꯛ",
            "how": "ꯀꯔꯝꯅ",
            
            # Responses
            "yes": "ꯍꯣꯏ",
            "no": "ꯅꯠꯇꯦ",
            "ok": "ꯑꯣꯛ",
            "okay": "ꯑꯣꯛ",
            "fine": "ꯊꯨꯅ",
            "well": "ꯐꯖꯅ"
        }
        
        # Split text into words and translate each word
        words = text_lower.split()
        translated_words = []
        
        for word in words:
            # Remove punctuation but keep it for later
            clean_word = re.sub(r'[^\w\s]', '', word)
            punctuation = re.sub(r'[\w\s]', '', word)
            
            if clean_word in translations:
                translated_word = translations[clean_word]
                # Add punctuation back if it existed
                if punctuation:
                    translated_word += punctuation
                translated_words.append(translated_word)
            else:
                # Keep unknown words as is
                translated_words.append(word)
        
        # Join translated words
        result = " ".join(translated_words)
        
        # Add common sentence patterns
        result = self._apply_sentence_patterns(result)
        
        # Capitalize first letter
        if result:
            result = result[0].upper() + result[1:]
        
        return result
    
    def _apply_sentence_patterns(self, text: str) -> str:
        """Apply common sentence patterns for better translation"""
        patterns = {
            r"ꯑꯩ ꯑꯣꯏ": "ꯑꯩꯅ ꯑꯣꯏ",
            r"ꯑꯗꯣꯝ ꯑꯣꯏ": "ꯑꯗꯣꯝꯅ ꯑꯣꯏ",
            r"ꯃꯍꯥꯛ ꯑꯣꯏ": "ꯃꯍꯥꯛꯅ ꯑꯣꯏ",
            r"ꯑꯩꯈꯣꯏ ꯑꯣꯏ": "ꯑꯩꯈꯣꯏꯅ ꯑꯣꯏ",
            r"ꯃꯈꯣꯏ ꯑꯣꯏ": "ꯃꯈꯣꯏꯅ ꯑꯣꯏ",
        }
        
        for pattern, replacement in patterns.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def translate_to_english(self, text: str) -> str:
        """Meitei Lon to English translation - simplified since we're removing this direction"""
        if not text or not text.strip():
            return ""
        
        # For now, just return a message since we're removing Meitei to English
        return "Meitei Lon to English translation is not available. Please use English to Meitei Lon translation."