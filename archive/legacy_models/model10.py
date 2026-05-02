from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model once (important for speed)
model_name = "facebook/nllb-200-distilled-600M"

print("Loading model... please wait")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Model loaded successfully")


def translate_en_to_dz(text):
    tokenizer.src_lang = "eng_Latn"

    inputs = tokenizer(text, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("dzo_Tibt"),
        max_length=100,
        num_beams=5,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Interactive loop
while True:
    user_input = input("\nEnter English text (or type 'exit' to quit): ")

    if user_input.lower() == "exit":
        print("Goodbye 👋")
        break

    translation = translate_en_to_dz(user_input)
    print("Dzongkha Translation:", translation)
