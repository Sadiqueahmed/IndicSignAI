# english.py - Centralized English Language Mappings and Utilities

import string

# CLASS_MAPPING: List of 178+ ISL signs recognized by the model
# Extracted from src/app.py
CLASS_MAPPING = [
    'A LOT', 'ABUSE', 'ALL', 'ANGRY', 'ANY', 'ANYTHING', 'APPRECIATE',
    'BEAUTIFUL', 'BED', 'BORED', 'BRING', 'CLASS', 'COLD', 'COLLEGE_SCHOOL', 'COMB',
    'COME', 'CRYING', 'DARE', 'DIFFERENCE', 'DILEMMA', 'DISAPPOINTED', 'DO', "DON'T CARE",
    'ENJOY', 'FAVOUR', 'FEVER', 'FINE', 'FOOD', 'FREE', 'FRIEND', 'GLASS', 'GO',
    'GOOD', 'GOT', 'GRATEFUL', 'HAD', 'HAPPENED', 'HAPPY', 'HEAR', 'HEART',
    'HELLO_HI', 'HELP', 'HIDING', 'HOW', 'HURT', 'I_ME_MINE_MY', 'KIND', 'KNOW',
    'LEAVE', 'LIGHT', 'LIKE', 'LIKE_LOVE', 'MAKE', 'MEAN IT', 'MEDICINE', 'NAME',
    'NEED', 'NEVER', 'NICE', 'NOT', 'NOW', 'NUMBER', 'OLD_AGE', 'ON THE WAY',
    'ONWARDS', 'OUTSIDE', 'PHONE', 'PLACE', 'PLANNED', 'POUR', 'PREPARE', 'PROMISE',
    'REALLY', 'REPEAT', 'ROOM', 'SERVE', 'SHIRT', 'SITTING', 'SLEEP', 'SLOWER',
    'SO MUCH', 'SOFTLY', 'SOME HOW', 'SOME MORE', 'SOME ONE', 'SOMETHING', 'SORRY',
    'SPEAK', 'STUBBORN', 'SURE', 'TAKE CARE', 'TAKE TIME', 'TALK', 'TELL', 'THANK',
    'THAT', 'THERE', 'THINGS', 'THINK', 'THIS ONE', 'TIRED', 'TRAIN', 'TRUST',
    'TRUTH', 'TURN ON', 'VERY', 'WANT', 'WATER', 'WEAR', 'WELCOME', 'WHAT', 'WHEN',
    'WHO', 'WORRY', 'afraid', 'again', 'agree', 'answer', 'assistance', 'attendance',
    'bad', 'become', 'book', 'break', 'careful', 'change', 'chat', 'college',
    'congratulations', 'doctor', 'email', 'file', 'from', 'good morning',
    'happy birthday', 'home', 'how are you', 'hungry', 'i need help', 'join',
    'keepsmile', 'meet', 'mistake', 'open', 'opinion', 'pain', 'pass', 'please',
    'practice', 'pray', 'pressure', 'problem', 'questions', 'remember', 'seat',
    'secondary', 'shift', 'sick', 'skin', 'small', 'specific', 'stand', 'stop',
    'sun', 'team', 'thirsty', 'this', 'today', 'together', 'understand', 'wait',
    'warn', 'where', 'which', 'work', 'write', 'you'
]

# FALLBACK_DICTIONARY: Translations for common English words/phrases
# Extracted from src/models/translation.py
FALLBACK_DICTIONARY = {
    # Greetings
    'hello': {
        'assamese': 'নমস্কাৰ', 'hindi': 'नमस्ते', 'manipuri': 'ꯑꯍꯥꯟꯕ',
        'nepali': 'नमस्ते', 'meitei_lon': 'ꯑꯍꯥꯟꯕ', 'marathi': 'नमस्कार',
        'tamil': 'வணக்கம்', 'bengali': 'হ্যালো', 'dzongkha': 'ཡ་ཨ་ན',
        'mizorami': 'Chibai', 'odia': 'ନମସ୍କাৰ'
    },
    'hi': {
        'assamese': 'নমস্কাৰ', 'hindi': 'नमस्ते', 'manipuri': 'ꯍꯥꯏ',
        'nepali': 'नमस्ते', 'meitei_lon': 'ꯍꯥꯏ', 'tamil': 'வணக்கம்',
        'dzongkha': 'ཡ་ཨ་ན'
    },
    'namaste': {
        'assamese': 'নমস্কাৰ', 'hindi': 'नमस्ते', 'manipuri': 'ꯅꯃꯁ꯭ꯇꯦ',
        'nepali': 'नमस्ते', 'meitei_lon': 'ꯅꯃꯁ꯭ꯇꯦ', 'dzongkha': 'ཞུ་གསོལ'
    },
    'thank you': {
        'assamese': 'ধন্যবাদ', 'hindi': 'धन्यवाद', 'manipuri': 'ꯊꯥꯠꯆꯔꯤ',
        'nepali': 'धन्यवाद', 'meitei_lon': 'ꯊꯥꯠꯆꯔꯤ', 'marathi': 'धन्यवाद',
        'tamil': 'நன்றி', 'bengali': 'ধন্যবাদ', 'dzongkha': 'བཀའ་དྲིན་ཆེ',
        'mizorami': 'Lawmthu', 'odia': 'ଧନ୍ୟବାଦ'
    },
    'good morning': {
        'assamese': 'শুভ ৰাতিপুৱা', 'hindi': 'শুভ প্রভাত', 'manipuri': 'ꯑꯍꯥꯟꯕ ꯑꯌꯨꯛ',
        'nepali': 'শুভ বিহান', 'meitei_lon': 'ꯑꯍꯥꯟꯕ ꯑꯌꯨꯛ', 'tamil': 'காலை வணக்கம்',
        'dzongkha': 'ང་ག་དྲོ་པ་བདེ་ལེགས', 'mizorami': 'Chibai', 'odia': 'ସୁପ୍ରଭାତ'
    },
    'how are you': {
        'assamese': 'আপোনাৰ কেনে আছে', 'hindi': 'आप कैसे हैं', 'manipuri': 'ꯑꯗꯣꯝ ꯀꯝꯗꯧꯔꯤ',
        'nepali': 'तपाईंलाई कस्तो छ', 'meitei_lon': 'ꯑꯗꯣꯝ ꯀꯝꯗꯧꯔꯤ', 'tamil': 'நீங்கள் எப்படி இருக்கிறீர்கள்',
        'dzongkha': 'ག་དེ་འབད་ཡོད་', 'mizorami': 'I dam em?', 'odia': 'କେମିତି ଅଛନ୍ତି'
    },
    'water': {
        'assamese': 'পানী', 'hindi': 'पानी', 'manipuri': 'ꯏꯁꯤꯡ',
        'nepali': 'पानी', 'meitei_lon': 'ꯏꯁꯤꯡ', 'tamil': 'தண்ணீர்',
        'dzongkha': 'ཆུ', 'mizorami': 'Tui', 'odia': 'ପାଣି'
    },
    'food': {
        'assamese': 'খাদ্য', 'hindi': 'भोजन', 'manipuri': 'ꯆꯥ',
        'nepali': 'खाना', 'meitei_lon': 'ꯆꯥ', 'tamil': 'உணவு',
        'dzongkha': 'ཟ་བ', 'mizorami': 'Ei', 'odia': 'ଖାଦ୍ୟ'
    },
    'help': {
        'assamese': 'সহায়', 'hindi': 'मदদ', 'manipuri': 'ꯃꯇꯦꯡ',
        'nepali': 'मद्दत', 'meitei_lon': 'ꯃꯇꯦꯡ', 'tamil': 'உதவி',
        'dzongkha': 'རོགས་པ', 'odia': 'ସହାୟତା'
    },
    'yes': {
        'assamese': 'হয়', 'hindi': 'हाँ', 'manipuri': 'ꯍꯣꯏ',
        'nepali': 'हो', 'meitei_lon': 'ꯍꯣꯏ', 'tamil': 'ஆம்',
        'dzongkha': 'ཨིན', 'odia': 'ହଁ'
    },
    'no': {
        'assamese': 'নহয়', 'hindi': 'नहीं', 'manipuri': 'ꯅꯠꯇꯦ',
        'nepali': 'होइन', 'meitei_lon': 'ꯅꯠꯇꯦ', 'tamil': 'இல்லை',
        'dzongkha': 'མེན', 'odia': 'ନାଁ'
    },
    'please': {
        'assamese': 'অনুগ্ৰহ কৰি', 'hindi': 'कृपया', 'manipuri': 'ꯆꯥꯟꯕꯤꯗꯨꯅꯥ',
        'nepali': 'कृपया', 'meitei_lon': 'ꯆꯥꯟꯕꯤꯗꯨꯅꯥ', 'tamil': 'தயவு செய்து',
        'dzongkha': 'གུ་མ་ཞུ་ག', 'odia': 'ଦୟାକରି'
    },
    'sorry': {
        'assamese': 'দুঃখিত', 'hindi': 'क्षमा करें', 'manipuri': 'ꯁꯣꯔꯤ',
        'nepali': 'माफ गर्नुहोस्', 'meitei_lon': 'ꯁꯣꯔꯤ', 'tamil': 'மன்னிக்கவும்',
        'dzongkha': 'དགོངས་མ་ཚོམ', 'odia': 'ଦୁଃଖିତ'
    }
}

def clean_text(text):
    """Clean and normalize English text"""
    if not text:
        return ""
    # Remove punctuation and convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.lower().strip()

def is_recognized_sign(text):
    """Check if a word is in the recognized ISL sign list"""
    return text.upper() in [s.upper() for s in CLASS_MAPPING]

def get_fallback_translation(text, target_lang):
    """Get manual translation from dictionary"""
    cleaned = clean_text(text)
    if cleaned in FALLBACK_DICTIONARY:
        return FALLBACK_DICTIONARY[cleaned].get(target_lang)
    return None


# ============================================================
# NLP SENTENCE CORRECTION ENGINE
# Rule-based ISL-to-English grammar correction
# ============================================================

# Pronoun / compound-sign normalization map
SIGN_NORMALIZATION = {
    'I_ME_MINE_MY':  'I',
    'HELLO_HI':      'hello',
    'COLLEGE_SCHOOL': 'school',
    'LIKE_LOVE':     'love',
    'OLD_AGE':       'old',
    "DON'T CARE":    "don't care",
    'A LOT':         'a lot',
    'MEAN IT':       'mean it',
    'ON THE WAY':    'on the way',
    'SO MUCH':       'so much',
    'SOME HOW':      'somehow',
    'SOME MORE':     'some more',
    'SOME ONE':      'someone',
    'THIS ONE':      'this one',
    'TAKE CARE':     'take care',
    'TAKE TIME':     'take time',
    'TURN ON':       'turn on',
}

# Words that are typically verbs in ISL context
ISL_VERBS = {
    'go', 'come', 'eat', 'drink', 'sleep', 'make', 'do', 'help', 'bring',
    'leave', 'like', 'love', 'want', 'need', 'know', 'think', 'speak',
    'talk', 'tell', 'hear', 'work', 'sit', 'stand', 'walk', 'run',
    'write', 'break', 'change', 'open', 'meet', 'join', 'wait', 'stop',
    'pray', 'practice', 'understand', 'wear', 'pour', 'prepare', 'serve',
    'trust', 'promise', 'repeat', 'train', 'enjoy', 'happen', 'plan',
    'hide', 'hurt', 'cry', 'worry', 'dare', 'appreciate', 'warn',
    'remember', 'agree', 'pass', 'become', 'chat',
}

# Words that are typically nouns/objects
ISL_NOUNS = {
    'home', 'school', 'college', 'food', 'water', 'medicine', 'phone',
    'place', 'room', 'bed', 'glass', 'shirt', 'book', 'name', 'number',
    'heart', 'light', 'sun', 'friend', 'doctor', 'team', 'email', 'file',
    'seat', 'skin', 'opinion', 'problem', 'question', 'pressure', 'pain',
    'mistake',
}

# Words that are adjectives/adverbs
ISL_MODIFIERS = {
    'good', 'happy', 'sad', 'angry', 'beautiful', 'nice', 'fine', 'tired',
    'bored', 'cold', 'free', 'grateful', 'disappointed', 'stubborn',
    'sorry', 'afraid', 'bad', 'careful', 'hungry', 'thirsty', 'sick',
    'small', 'specific', 'very', 'really', 'never', 'softly', 'slower',
    'kind',
}

# Common ISL sentence pattern templates (SOV → SVO etc.)
ISL_PHRASE_PATTERNS = {
    # (normalised input tuple) → corrected English
    ('I', 'home', 'go'):          'I am going home',
    ('I', 'go', 'home'):          'I am going home',
    ('I', 'food', 'want'):        'I want food',
    ('I', 'want', 'food'):        'I want food',
    ('I', 'water', 'want'):       'I want water',
    ('I', 'want', 'water'):       'I want water',
    ('I', 'help', 'need'):        'I need help',
    ('I', 'need', 'help'):        'I need help',
    ('I', 'fine'):                'I am fine',
    ('I', 'happy'):               'I am happy',
    ('I', 'sad'):                 'I am sad',
    ('I', 'tired'):               'I am tired',
    ('I', 'hungry'):              'I am hungry',
    ('I', 'thirsty'):             'I am thirsty',
    ('I', 'sick'):                'I am sick',
    ('I', 'sorry'):               'I am sorry',
    ('how', 'you'):               'How are you?',
    ('you', 'how'):               'How are you?',
    ('what', 'name'):             'What is your name?',
    ('name', 'what'):             'What is your name?',
    ('thank', 'you'):             'Thank you',
    ('I', 'school', 'go'):        'I am going to school',
    ('I', 'go', 'school'):        'I am going to school',
    ('I', 'college', 'go'):       'I am going to college',
    ('I', 'go', 'college'):       'I am going to college',
    ('you', 'good'):              'You are good',
    ('you', 'beautiful'):         'You are beautiful',
    ('I', 'love', 'you'):         'I love you',
    ('I', 'you', 'love'):         'I love you',
    ('I', 'know'):                'I know',
    ('I', 'understand'):          'I understand',
    ('please', 'help'):           'Please help me',
    ('I', 'eat'):                 'I am eating',
    ('I', 'sleep'):               'I am sleeping',
    ('I', 'drink', 'water'):      'I am drinking water',
    ('I', 'go'):                  'I am going',
    ('I', 'come'):                'I am coming',
    ('I', 'work'):                'I am working',
    ('good', 'morning'):          'Good morning',
    ('happy', 'birthday'):        'Happy birthday',
}


def normalize_sign(sign_text):
    """Normalize a raw ISL sign label to a clean English word."""
    s = sign_text.strip()
    # Check exact match in normalization map (case-insensitive)
    upper = s.upper()
    if upper in SIGN_NORMALIZATION:
        return SIGN_NORMALIZATION[upper]
    # Return lowercase version
    return s.lower()


def correct_sentence(words):
    """
    Convert a list of raw ISL sign labels into a fluent English sentence.
    
    Process:
      1. Normalize each sign label
      2. Try known phrase-pattern matching
      3. Apply grammar rules (SOV→SVO, articles, tense)
      4. Capitalize and punctuate
    
    Args:
        words: list of raw sign strings, e.g. ['I_ME_MINE_MY', 'HOME', 'GO']
    
    Returns:
        str: corrected English sentence, e.g. 'I am going home.'
    """
    if not words:
        return ""
    
    # Step 1: Normalize
    normalized = [normalize_sign(w) for w in words]
    # Remove duplicates that appear consecutively (common in ISL detection)
    deduped = [normalized[0]]
    for w in normalized[1:]:
        if w != deduped[-1]:
            deduped.append(w)
    normalized = deduped
    
    # Step 2: Try exact phrase-pattern match
    key = tuple(normalized)
    if key in ISL_PHRASE_PATTERNS:
        result = ISL_PHRASE_PATTERNS[key]
        if not result.endswith(('?', '!', '.')):
            result += '.'
        return result
    
    # Step 3: Apply grammar rules
    result = _apply_grammar_rules(normalized)
    
    # Step 4: Capitalize first letter, add period if needed
    if result:
        result = result[0].upper() + result[1:]
        if not result.endswith(('?', '!', '.')):
            result += '.'
    
    return result


def _apply_grammar_rules(words):
    """Apply ISL-to-English grammar transformations."""
    if not words:
        return ""
    
    result = list(words)  # work on a mutable copy
    
    # --- Rule 1: If subject is 'I' and next word is adjective, insert 'am' ---
    for i in range(len(result) - 1):
        if result[i].lower() == 'i' and result[i+1].lower() in ISL_MODIFIERS:
            result.insert(i + 1, 'am')
            break
    
    # --- Rule 2: If subject is 'you' and next word is adjective, insert 'are' ---
    for i in range(len(result) - 1):
        if result[i].lower() == 'you' and result[i+1].lower() in ISL_MODIFIERS:
            result.insert(i + 1, 'are')
            break
    
    # --- Rule 3: SOV → SVO reordering ---
    # If pattern is [Subject] [Object/Noun] [Verb], swap last two
    if len(result) >= 3:
        subj = result[0].lower()
        mid = result[-2].lower()
        last = result[-1].lower()
        if subj in ('i', 'you', 'we', 'they', 'he', 'she') and \
           mid in ISL_NOUNS and last in ISL_VERBS:
            result[-2], result[-1] = result[-1], result[-2]
    
    # --- Rule 4: Progressive tense for lone verbs after subject ---
    for i in range(len(result) - 1):
        subj = result[i].lower()
        if subj in ('i', 'you', 'we', 'they', 'he', 'she'):
            next_w = result[i+1].lower()
            if next_w in ISL_VERBS:
                # Check if there's already a 'be' verb
                if i + 1 < len(result) and result[i+1].lower() not in ('am', 'are', 'is'):
                    be_verb = 'am' if subj == 'i' else 'are' if subj in ('you', 'we', 'they') else 'is'
                    # Add progressive -ing
                    verb = result[i+1].lower()
                    ing_form = _to_progressive(verb)
                    result[i+1] = ing_form
                    result.insert(i+1, be_verb)
                break
    
    # --- Rule 5: Add article before lone nouns if needed ---
    for i in range(len(result)):
        w = result[i].lower()
        if w in ISL_NOUNS:
            # Don't add article if preceded by a pronoun/possessive or another article
            if i == 0 or result[i-1].lower() not in ('a', 'an', 'the', 'my', 'your',
                                                       'his', 'her', 'our', 'their',
                                                       'some', 'this', 'that'):
                # Don't add if it's preceded by a preposition context that already flows
                prev = result[i-1].lower() if i > 0 else ''
                if prev not in ('to', 'at', 'in', 'on', 'for', 'from', 'with'):
                    pass  # Could insert 'the' but ISL often implies definite context
    
    # --- Rule 6: Question words should move to front ---
    question_words = {'what', 'who', 'where', 'when', 'how', 'which', 'why'}
    for i in range(1, len(result)):
        if result[i].lower() in question_words:
            q_word = result.pop(i)
            result.insert(0, q_word)
            # Append '?' at the end
            break
    
    # Check if result starts with a question word
    if result and result[0].lower() in question_words:
        sentence = ' '.join(result)
        if not sentence.endswith('?'):
            sentence += '?'
        return sentence
    
    return ' '.join(result)


def _to_progressive(verb):
    """Convert a base verb to -ing form."""
    # Common irregular forms
    irregulars = {
        'go': 'going', 'come': 'coming', 'make': 'making',
        'have': 'having', 'give': 'giving', 'take': 'taking',
        'leave': 'leaving', 'hide': 'hiding', 'write': 'writing',
        'break': 'breaking', 'change': 'changing', 'practice': 'practicing',
        'prepare': 'preparing', 'serve': 'serving', 'believe': 'believing',
        'die': 'dying', 'lie': 'lying', 'tie': 'tying',
        'run': 'running', 'sit': 'sitting', 'stop': 'stopping',
        'eat': 'eating', 'sleep': 'sleeping', 'drink': 'drinking',
        'speak': 'speaking', 'talk': 'talking', 'tell': 'telling',
        'think': 'thinking', 'work': 'working', 'walk': 'walking',
        'stand': 'standing', 'cry': 'crying', 'try': 'trying',
        'wear': 'wearing', 'pour': 'pouring', 'train': 'training',
        'enjoy': 'enjoying', 'pray': 'praying', 'worry': 'worrying',
        'help': 'helping', 'need': 'needing', 'want': 'wanting',
        'know': 'knowing', 'understand': 'understanding',
    }
    if verb in irregulars:
        return irregulars[verb]
    
    # Rule-based: silent-e → drop e, add -ing
    if verb.endswith('e') and not verb.endswith('ee'):
        return verb[:-1] + 'ing'
    # CVC doubling for short verbs
    if len(verb) <= 4 and verb[-1] not in 'aeiouwxy' and verb[-2] in 'aeiou':
        return verb + verb[-1] + 'ing'
    return verb + 'ing'
