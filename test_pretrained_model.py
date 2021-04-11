# -*- coding: utf-8 -*-

from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('gpt2') 
model = AutoModelForCausalLM.from_pretrained('gpt2',pad_token_id=tokenizer.eos_token_id)

gpt2_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1,)

generated_text = gpt2_pipeline("I live to")[0]["generated_text"]
print(generated_text)
