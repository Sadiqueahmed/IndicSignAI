def correct_sentence(words):
    """
    NLP Grammar Correction.
    Takes a list of words (ISL glosses) and returns a grammatically 
    smoothed English sentence.
    """
    if not words:
        return ""
    
    # Very basic smoothing for the prototype:
    # Join with spaces, capitalize first letter, add period.
    sentence = " ".join(words).strip()
    
    # Capitalize first character
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
    
    return sentence
