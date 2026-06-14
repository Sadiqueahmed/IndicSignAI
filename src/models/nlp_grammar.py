"""
NLP Grammar Correction Module.
Routes to the full ISL-to-English grammar engine in english.py.
"""
import os
import sys

# Ensure the project root is importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import the full grammar engine from english.py at project root
try:
    from english import correct_sentence as _english_correct_sentence
    _HAS_ENGLISH_MODULE = True
except ImportError:
    _HAS_ENGLISH_MODULE = False


def correct_sentence(words):
    """
    NLP Grammar Correction.
    Takes a list of words (ISL glosses) and returns a grammatically
    smoothed English sentence.
    
    Delegates to english.py's full rule-based engine which handles:
      - Sign normalization (I_ME_MINE_MY → I, HELLO_HI → hello, etc.)
      - Known phrase-pattern matching (ISL SOV → English SVO)
      - Grammar rules (progressive tense, articles, question reordering)
      - Consecutive duplicate removal
      - Proper capitalization and punctuation
    
    Args:
        words: list of raw sign strings, e.g. ['I_ME_MINE_MY', 'GO', 'HOME']
    
    Returns:
        str: corrected English sentence, e.g. 'I am going home.'
    """
    if not words:
        return ""
    
    if _HAS_ENGLISH_MODULE:
        return _english_correct_sentence(words)
    
    # Fallback: basic join + capitalize if english.py is not available
    sentence = " ".join(words).strip()
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
    return sentence
