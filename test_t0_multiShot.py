import CaptionDataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B", cache_dir='./models')
model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B", cache_dir='./models')

inputs = tokenizer("Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy", max_length=512, truncation=True, padding=True, return_tensors="pt")

outputs = model.generate(**inputs)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
