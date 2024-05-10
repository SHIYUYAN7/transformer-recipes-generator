def generate_recipe(src_text, model, tokenizer):
    input_ids = tokenizer(src_text, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_length=1000, num_beams=4, no_repeat_ngram_size=2, early_stopping=False)
    return tokenizer.decode(output[0], skip_special_tokens=True)
