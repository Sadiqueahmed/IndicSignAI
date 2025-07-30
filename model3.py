from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load model and tokenizer
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Define translation pipeline for English to Nepali
translator = pipeline(
    "translation",
    model=model,
    tokenizer=tokenizer,
    src_lang="eng_Latn",   # English (Latin script)
    tgt_lang="npi_Deva"    # Nepali (Devanagari script)
)

# Interactive translation loop
print("English ➝ Nepali Translator")
print("Type 'exit' to quit.\n")

while True:
    english_text = input("Enter English text: ")
    if english_text.lower() == "exit":
        break

    try:
        result = translator(english_text)
        print("Nepali Translation:", result[0]["translation_text"], "\n")
    except Exception as e:
        print("Error:", str(e), "\n")
