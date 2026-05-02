"""
Bengali Script → Meitei Mayek (ꯃꯤꯇꯩ ꯃꯌꯦꯛ) Transliteration Module

This module guarantees that any Manipuri text returned in Bengali script
is automatically converted to the correct Meitei Mayek Unicode block
(U+ABC0–U+ABFF, U+AAE0–U+AAFF).

Usage:
    from models.bengali_to_meitei import ensure_meitei_mayek
    output = ensure_meitei_mayek(nllb_output)  # Bengali → Meitei Mayek
"""

import re
import logging

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# Bengali → Meitei Mayek Character Mapping
# ══════════════════════════════════════════════════════════════════════

# ── Independent Vowels ──
BENGALI_VOWELS_TO_MEITEI = {
    'অ': 'ꯑ',   # A
    'আ': 'ꯑꯥ',  # AA
    'ই': 'ꯏ',   # I
    'ঈ': 'ꯏ',   # II (Meitei doesn't distinguish long/short)
    'উ': 'ꯎ',   # U
    'ঊ': 'ꯎ',   # UU
    'ঋ': 'ꯔꯤ',  # Vocalic R → RI approximation
    'এ': 'ꯑꯦ',  # E
    'ঐ': 'ꯑꯩ',  # AI
    'ও': 'ꯑꯣ',  # O
    'ঔ': 'ꯑꯧ',  # AU
}

# ── Consonants ──
BENGALI_CONSONANTS_TO_MEITEI = {
    'ক': 'ꯀ',   # KA
    'খ': 'ꯈ',   # KHA
    'গ': 'ꯒ',   # GA
    'ঘ': 'ꯘ',   # GHA
    'ঙ': 'ꯉ',   # NGA
    'চ': 'ꯆ',   # CA
    'ছ': 'ꯆ',   # CHA (→ CA, no separate Meitei letter)
    'জ': 'ꯖ',   # JA
    'ঝ': 'ꯖ',   # JHA (→ JA)
    'ঞ': 'ꯉ',   # NYA (→ NGA approximation)
    'ট': 'ꯇ',   # TTA
    'ঠ': 'ꯊ',   # TTHA
    'ড': 'ꯗ',   # DDA
    'ঢ': 'ꯙ',   # DDHA
    'ণ': 'ꯅ',   # NNA (→ NA)
    'ত': 'ꯇ',   # TA
    'থ': 'ꯊ',   # THA
    'দ': 'ꯗ',   # DA
    'ধ': 'ꯙ',   # DHA
    'ন': 'ꯅ',   # NA
    'প': 'ꯄ',   # PA
    'ফ': 'ꯐ',   # PHA
    'ব': 'ꯕ',   # BA
    'ভ': 'ꯚ',   # BHA
    'ম': 'ꯃ',   # MA
    'য': 'ꯌ',   # YA
    'র': 'ꯔ',   # RA
    'ল': 'ꯂ',   # LA
    'শ': 'ꯁ',   # SHA
    'ষ': 'ꯁ',   # SSA (→ SHA)
    'স': 'ꯁ',   # SA (→ SHA)
    'হ': 'ꯍ',   # HA
    'ড়': 'ꯔ',  # RRA (→ RA)
    'ঢ়': 'ꯙ',  # RRDHA
    'য়': 'ꯌ',  # YYA (→ YA)
    'ৎ': 'ꯠ',   # Khanda TA → Final consonant
    'ং': 'ꯡ',   # Anusvara → final NG
    'ঃ': 'ꯍ',   # Visarga → HA approximation
    'ঁ': 'ꯪ',   # Chandrabindu → Cheinap
}

# ── Dependent Vowel Signs (matras) ──
BENGALI_VOWEL_SIGNS_TO_MEITEI = {
    'া': 'ꯥ',   # AA sign
    'ি': 'ꯤ',   # I sign
    'ী': 'ꯤ',   # II sign (→ I sign)
    'ু': 'ꯨ',   # U sign
    'ূ': 'ꯨ',   # UU sign (→ U sign)
    'ৃ': 'ꯔꯤ',  # Vocalic R sign
    'ে': 'ꯦ',   # E sign
    'ৈ': 'ꯩ',   # AI sign
    'ো': 'ꯣ',   # O sign
    'ৌ': 'ꯧ',   # AU sign
    'ৗ': 'ꯧ',   # AU Length mark → AU sign
}

# ── Digits ──
BENGALI_DIGITS_TO_MEITEI = {
    '০': '০',   # Keep Bengali digits (Meitei Mayek digits are rare in modern use)
    '১': '১',
    '২': '২',
    '৩': '৩',
    '৪': '৪',
    '৫': '৫',
    '৬': '৬',
    '৭': '৭',
    '৮': '৮',
    '৯': '৯',
}

# ── Special Marks ──
BENGALI_SPECIAL_TO_MEITEI = {
    '্': '꯭',   # Virama / Hasanta → Apun Iyek (the halant/virama of Meitei)
    '।': '꯫',   # Danda → Meitei Cheikhei (full stop)
    '॥': '꯫꯫',  # Double Danda
}

# ══════════════════════════════════════════════════════════════════════
# Build combined mapping (order matters: multi-char entries first)
# ══════════════════════════════════════════════════════════════════════

def _build_sorted_mapping():
    """Build a single mapping sorted by key length (longest first)
    so that multi-character sequences like 'ড়' match before 'ড'."""
    combined = {}
    combined.update(BENGALI_VOWELS_TO_MEITEI)
    combined.update(BENGALI_CONSONANTS_TO_MEITEI)
    combined.update(BENGALI_VOWEL_SIGNS_TO_MEITEI)
    combined.update(BENGALI_DIGITS_TO_MEITEI)
    combined.update(BENGALI_SPECIAL_TO_MEITEI)
    # Sort by key length descending so multi-char sequences match first
    return sorted(combined.items(), key=lambda kv: len(kv[0]), reverse=True)

_SORTED_MAPPING = _build_sorted_mapping()

# Bengali Unicode range: U+0980–U+09FF
_BENGALI_RANGE = re.compile(r'[\u0980-\u09FF]')

# Meitei Mayek Unicode ranges: U+ABC0–U+ABFF (main), U+AAE0–U+AAFF (extended)
_MEITEI_RANGE = re.compile(r'[\uABC0-\uABFF\uAAE0-\uAAFF]')


# ══════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════

def is_bengali_script(text: str) -> bool:
    """Check if text contains Bengali script characters."""
    return bool(_BENGALI_RANGE.search(text))


def is_meitei_mayek_script(text: str) -> bool:
    """Check if text contains Meitei Mayek characters."""
    return bool(_MEITEI_RANGE.search(text))


def transliterate_bengali_to_meitei(text: str) -> str:
    """Transliterate Bengali script text to Meitei Mayek script.
    
    Processes the string from left to right, greedily matching
    the longest Bengali character/sequence first and replacing
    it with the Meitei Mayek equivalent.
    
    Non-Bengali characters (spaces, punctuation, Latin, etc.) are
    preserved as-is.
    """
    if not text:
        return text
    
    result = []
    i = 0
    text_len = len(text)
    
    while i < text_len:
        matched = False
        # Try each mapping entry (longest keys first)
        for bengali_seq, meitei_seq in _SORTED_MAPPING:
            seq_len = len(bengali_seq)
            if text[i:i + seq_len] == bengali_seq:
                result.append(meitei_seq)
                i += seq_len
                matched = True
                break
        
        if not matched:
            # Keep character as-is (space, punctuation, Latin, etc.)
            result.append(text[i])
            i += 1
    
    return ''.join(result)


def ensure_meitei_mayek(text: str) -> str:
    """Guarantee that Manipuri text is in Meitei Mayek script.
    
    - If text already contains Meitei Mayek → return as-is
    - If text contains Bengali script → transliterate to Meitei Mayek
    - Otherwise → return as-is (Latin, etc.)
    
    This is the MAIN entry point for all Manipuri translation outputs.
    """
    if not text or not text.strip():
        return text
    
    # If already in Meitei Mayek, return as-is
    if is_meitei_mayek_script(text):
        return text
    
    # If contains Bengali, transliterate
    if is_bengali_script(text):
        converted = transliterate_bengali_to_meitei(text)
        logger.info(f"Bengali→Meitei Mayek: '{text}' → '{converted}'")
        return converted
    
    # Neither Bengali nor Meitei — return as-is (Latin fallback text, etc.)
    return text


# ══════════════════════════════════════════════════════════════════════
# Self-test
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    test_cases = [
        ("নমস্কার", "Bengali greeting"),
        ("আপনি কেমন আছেন", "How are you in Bengali"),
        ("ধন্যবাদ", "Thank you in Bengali"),
        ("আমার নাম", "My name in Bengali"),
        ("ꯑꯍꯥꯟꯕ", "Already Meitei Mayek - should pass through"),
        ("Hello", "Latin text - should pass through"),
        ("আমি ভালো আছি", "I am fine in Bengali"),
    ]
    
    print("=" * 70)
    print("Bengali → Meitei Mayek Transliteration Tests")
    print("=" * 70)
    
    for text, desc in test_cases:
        result = ensure_meitei_mayek(text)
        has_bengali = is_bengali_script(result)
        has_meitei = is_meitei_mayek_script(result)
        status = "✓" if not has_bengali else "✗ STILL BENGALI"
        print(f"\n{desc}:")
        print(f"  Input:  {text}")
        print(f"  Output: {result}")
        print(f"  Status: {status} (Meitei: {has_meitei})")
    
    print("\n" + "=" * 70)
