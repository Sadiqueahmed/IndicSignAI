# meitei_transliterator.py
import re

class MeiteiTransliterator:
    def __init__(self):
        # Mapping from Bengali script to Meitei Mayek
        self.bengali_to_meitei = {
            # Vowels
            'অ': 'ꯑ', 'আ': 'ꯑ', 'ই': 'ꯏ', 'ঈ': 'ꯏ', 'উ': 'ꯎ', 'ঊ': 'ꯎ',
            'এ': 'ꯑꯩ', 'ঐ': 'ꯑꯩ', 'ও': 'ꯑꯧ', 'ঔ': 'ꯑꯧ',
            
            # Consonants
            'ক': 'ꯀ', 'খ': 'ꯈ', 'গ': 'ꯒ', 'ঘ': 'ꯘ', 'ঙ': 'ꯉ',
            'চ': 'ꯆ', 'ছ': 'ꯆ', 'জ': 'ꯖ', 'ঝ': 'ꯓ', 'ঞ': 'ꯜ',
            'ট': 'ꯇ', 'ঠ': 'ꯊ', 'ড': 'ꯗ', 'ঢ': 'ꯙ', 'ণ': 'ꯟ',
            'ত': 'ꯇ', 'থ': 'ꯊ', 'দ': 'ꯗ', 'ধ': 'ꯙ', 'ন': 'ꯅ',
            'প': 'ꯄ', 'ফ': 'ꯐ', 'ব': 'ꯕ', 'ভ': 'ꯚ', 'ম': 'ꯃ',
            'য': 'ꯌ', 'র': 'ꯔ', 'ল': 'ꯂ', 'ৱ': 'ꯋ',
            'শ': 'ꯁ', 'ষ': 'ꯁ', 'স': 'ꯁ', 'হ': 'ꯍ',
            
            # Diacritics and modifiers
            'া': 'ꯥ', 'ি': 'ꯤ', 'ী': 'ꯤ', 'ু': 'ꯨ', 'ূ': 'ꯨ',
            'ে': 'ꯦ', 'ৈ': 'ꯦ', 'ো': 'ꯣ', 'ৌ': 'ꯣ',
            '্': '', 'ং': 'ꯡ', 'ঃ': 'ꯡ',
            
            # Numbers
            '০': '꯰', '১': '꯱', '২': '꯲', '৩': '꯳', '৪': '꯴',
            '৫': '꯵', '৬': '꯶', '৭': '꯷', '৮': '꯸', '৯': '꯹'
        }
        
        # Common word mappings for better accuracy
        self.special_mappings = {
            'মনিপুর': 'ꯃꯅꯤꯄꯨꯔ',
            'মণিপুর': 'ꯃꯅꯤꯄꯨꯔ',
            'কঙ্গলেই': 'ꯀꯪꯒꯂꯩ',
            'ইম্ফল': 'ꯏꯝꯐꯥꯜ',
            'নং': 'ꯅꯪ',
            'হাই': 'ꯍꯥꯏ',
            'খার': 'ꯈꯥꯔ',
            'লৌ': 'ꯂꯧ',
            'চিং': 'ꯆꯤꯡ',
            'থৌ': 'ꯊꯧ'
        }

    def transliterate_bengali_to_meitei(self, text):
        """Convert Bengali script text to Meitei Mayek script"""
        if not text:
            return text
            
        # Convert to string if needed
        text = str(text)
        
        # First apply special word mappings
        for bengali_word, meitei_word in self.special_mappings.items():
            text = text.replace(bengali_word, meitei_word)
        
        # Then transliterate character by character
        result = []
        i = 0
        while i < len(text):
            char = text[i]
            
            # Handle combined characters
            if i + 1 < len(text):
                two_chars = text[i:i+2]
                if two_chars in ['ং', 'ঃ', 'ৈ', 'ৌ']:
                    if two_chars in self.bengali_to_meitei:
                        result.append(self.bengali_to_meitei[two_chars])
                    i += 2
                    continue
            
            # Handle single character
            if char in self.bengali_to_meitei:
                result.append(self.bengali_to_meitei[char])
            else:
                result.append(char)  # Keep non-Bengali characters as-is
            i += 1
        
        return ''.join(result)
    
    def is_bengali_script(self, text):
        """Check if text contains Bengali script characters"""
        bengali_range = r'[\u0980-\u09FF]'
        return bool(re.search(bengali_range, text))

# Test function
def test_transliterator():
    transliterator = MeiteiTransliterator()
    
    test_samples = [
        "মনিপুর",
        "কঙ্গলেই",
        "হেলো",
        "ধন্যবাদ",
        "নং",
        "খার",
        "চিং"
    ]
    
    print("Testing Meitei Mayek transliteration:")
    for sample in test_samples:
        converted = transliterator.transliterate_bengali_to_meitei(sample)
        print(f"{sample} -> {converted}")

if __name__ == "__main__":
    test_transliterator()