from datasets import load_dataset
from tokenizer import CheessTokenizer
from torch import torch
import numpy as np
from tqdm import tqdm
def process(example,tokenizer):
        column_name = "transcript"
        split_text = example[column_name].split(";")
        #print(split_text)
        first = split_text[1]
        ids = np.array([tokenizer.encode(first)])
        out = {"ids": ids, "len": len(ids)}
        return out
def pretokenize_dataset(dataset_name,file_name,tokenizer):
    num_processes = 64
    original_dataset = load_dataset(dataset_name, data_files=file_name)["train"]
    split = original_dataset.train_test_split(test_size=0.2,seed=42)
    tokenized = split.map(process,fn_kwargs={"tokenizer":tokenizer},remove_columns="transcript",num_proc=num_processes)
    train = tokenized["train"]
    train.save_to_disk("train_dataset"+"_"+file_name)
    test = tokenized["test"]
    test.save_to_disk("test_dataset"+"_"+file_name)    
    #return tokenized


dataset_name="adamkarvonen/chess_games"
file_name="stockfish_dataset_blocks.zip"
pretokenize_dataset(dataset_name,file_name,CheessTokenizer())