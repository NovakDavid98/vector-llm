import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer

def get_wikitext_dataset(tokenizer_name="gpt2", block_size=1024):
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=block_size)
    
    print("Tokenizing...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # We need to flatten and chunk
    # But for a simple fine-tuning loop, let's just use the tokenized texts that are long enough
    # or implement the standard "group_texts" from HF examples.
    
    # Simplified approach: Filter out empty lines
    tokenized_datasets = tokenized_datasets.filter(lambda x: len(x["input_ids"]) > 0)
    
    return tokenized_datasets, tokenizer
