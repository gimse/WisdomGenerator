# -*- coding: utf-8 -*-

import csv
from transformers import AutoTokenizer
import random

tokenizer = AutoTokenizer.from_pretrained('gpt2')

count=0
line_count=0
train_m = ""
test_m = ""

with open('quotes_dataset.csv',encoding="utf8") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if line_count%1000==0:print(line_count)
        line_count=line_count+1
        tags=row[2].split(",")
        for tag in tags:
            tag=tag.strip()
            if tag in ["love", "life", "inspirational", "motivational", "life lessons", "dreams"]:
                count=count+1
                line=row[0]
                if len(line.split())>100: continue
               
                if random.random()<0.9: train_m += (tokenizer.special_tokens_map['bos_token']+line.rstrip()+tokenizer.special_tokens_map['eos_token'])
                else:  test_m+=(tokenizer.special_tokens_map['bos_token']+line.rstrip()+tokenizer.special_tokens_map['eos_token'])
                break

train_path='./data/train.txt'
test_path='./data/test.txt'

with open(train_path, "w", encoding='utf-8') as f:
    f.write(train_m)
    
with open(test_path, "w", encoding='utf-8') as f:
    f.write(test_m)