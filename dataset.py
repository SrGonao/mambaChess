from datasets import load_dataset
from torch import torch
import numpy as np
from tqdm import tqdm
from tokenizer import ChessSimpleTokenizer,ChessTokenizer

def tokenize(example,tokenizer):
        column_name = "transcript"
        split_text = example[column_name].split(";")
        #print(split_text)
        first = split_text[1]
        ids = tokenizer.encode(first)   
        decode = tokenizer.decode(ids)
        if len(decode)>len(first):
            decode = decode[:-1]
        try:
            assert first == decode
        except:
            print("Skipped")
            print(first,decode)
            print("############")
            ids = []
        out = {"ids": ids, "len": len(ids)}
        return out

def decode(example,tokenizer):
    column_name = "train"
    ids = example[column_name]["ids"]
    return tokenizer.decode(ids)

def pretokenize_dataset(dataset_name,file_name,tokenizer,name="simple"):
    num_processes = 64
    original_dataset = load_dataset(dataset_name, data_files=file_name)["train"]
    split = original_dataset.train_test_split(test_size=0.2,seed=42)
    tokenized = split.map(tokenize,fn_kwargs={"tokenizer":tokenizer},remove_columns="transcript",num_proc=num_processes)
    train = tokenized["train"]

    train.save_to_disk(f"datasets/{name}_train_dataset"+"_"+file_name)
    test = tokenized["test"]
    test.save_to_disk(f"datasets/{name}_test_dataset"+"_"+file_name)    
    #return tokenized

def test_tokenization(dataset_name,file_name,tokenizer):
    dataset = load_dataset(dataset_name, data_files=file_name)
    dataset = dataset["train"]
    examples = 0
    for example in dataset:
        split_text = example["transcript"].split(";")
        first = split_text[1]
        ids = tokenizer.encode(first)
        decoded = tokenizer.decode(ids)
        if len(decoded)>len(first):
            decoded = decoded[:-1]

        try: 
             assert first == decoded
        except:    
            print(f"##################### {examples} ######################")
            print(first)
            print("\n")
            print(decoded)
            exit()
        examples+=1


dataset_name="adamkarvonen/chess_games"
file_name="stockfish_dataset_blocks.zip"
simple_tokenizer = ChessSimpleTokenizer()
normal_tokenizer = ChessTokenizer()

#test_tokenization(dataset_name,file_name,normal_tokenizer)
pretokenize_dataset(dataset_name,file_name,simple_tokenizer,name="simple")
pretokenize_dataset(dataset_name,file_name,normal_tokenizer,name="normal")
