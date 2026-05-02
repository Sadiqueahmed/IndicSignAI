from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
translator = pipeline(
    "translation",
    model=model,
    tokenizer=tokenizer,
    src_lang="eng_Latn",
    tgt_lang="asm_Beng"
)

print("English ➝ Assamese Translator")
print("Type 'exit' to quit.\n")

while True:
    english_text = input("Enter English text: ")
    if english_text.lower() == "exit":
        break

    try:
        result = translator(english_text)
        print("Assamese Translation:", result[0]["translation_text"], "\n")
    except Exception as e:
        print("Error:", str(e), "\n")
