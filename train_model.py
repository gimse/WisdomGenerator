# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, Trainer, TrainingArguments, DataCollatorForLanguageModeling

tokenizer = AutoTokenizer.from_pretrained('gpt2') 
model = AutoModelForCausalLM.from_pretrained('gpt2',pad_token_id=tokenizer.eos_token_id)

training_args = TrainingArguments(
    output_dir="./storage/gpt2-motivational_v6", #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=10, # number of training epochs
    per_gpu_train_batch_size=32, # batch size for training
    per_gpu_eval_batch_size=64,  # batch size for evaluation
    logging_steps = 500, # Number of update steps between two evaluations.
    save_steps=500, # after # steps model is saved
    warmup_steps=500,# number of warmup steps for learning rate scheduler
    )

train_path='./data/train.txt'
test_path='./data/test.txt'

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
