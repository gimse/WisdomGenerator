# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, Trainer, TrainingArguments, DataCollatorForLanguageModeling

tokenizer = AutoTokenizer.from_pretrained('gpt2') 
model = AutoModelForCausalLM.from_pretrained('gpt2',pad_token_id=tokenizer.eos_token_id)

training_args = TrainingArguments(
    output_dir="./saved_models/gpt2-wistomwords", #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=1, # number of training epochs
    )

train_path='./dummy_data/train_dummy.txt'
test_path='./dummy_data/test_dummy.txt'

train_dataset=TextDataset(tokenizer=tokenizer,file_path=train_path, block_size=1)
test_dataset=TextDataset(tokenizer=tokenizer,file_path=test_path,block_size=1)
data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False,)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
trainer.save_model("./saved_models/gpt2-wistomwords_v1")
