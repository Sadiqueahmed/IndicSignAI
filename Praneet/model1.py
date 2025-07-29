from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load model and tokenizer
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Define translation pipeline for English to Hindi
translator = pipeline(
    "translation",
    model=model,
    tokenizer=tokenizer,
    src_lang="eng_Latn",   # Source: English (Latin script)
    tgt_lang="hin_Deva"    # Target: Hindi (Devanagari script)
)

# Interactive translator
print("English ➝ Hindi Translator")
print("Type 'exit' to quit.\n")

while True:
    english_text = input("Enter English text: ")
    if english_text.lower() == "exit":
        break

    try:
        result = translator(english_text)
        print("Hindi Translation:", result[0]["translation_text"], "\n")
    except Exception as e:
        print("Error:", str(e), "\n")
