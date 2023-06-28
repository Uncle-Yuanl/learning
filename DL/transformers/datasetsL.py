from functools import partial
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples, tokenizer):
    return tokenizer(examples["sentence"], padding="max_length", max_length=128, truncation=True)

# huggingface dataset result
dataset = load_dataset(
    path="glue",
    name="sst2",
    split="train",
)

hdataset = dataset.map(partial(preprocess_function, tokenizer=tokenizer))
"""
dataset[0]
{
    'sentence': 'hide new secretions ...tal units ', 
    'label': 0, 'idx': 0, 
    'input_ids': [101, 5342, 2047, 3595, 8496, 2013, 1996, 18643, 3197, ...], 
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, ...]
}
"""

# local file 
ldataset = load_dataset(
    path="json",
    data_files={
        'train': "/home/yhao/pretrained_models/concept/nutrition/search/Acceptable_Costs/data/train.jsonl",
        'valid': "/home/yhao/pretrained_models/concept/nutrition/search/Acceptable_Costs/data/valid.jsonl"
    }
)
print()
