from tokenizer import ChessSimpleTokenizer,ChessTokenizer
from model import ChessMambaModel, ChessMambaConfig, ChessLlamaModel, ChessLlamaConfig
from datasets import load_from_disk
import torch
from tqdm import tqdm
import wandb
import os
from torch.optim import AdamW
from transformers import Trainer, TrainingArguments,LlamaConfig
import argparse
import glob


#Train params
#if torch.cuda.is_available():
#    device = torch.device("cuda:7")


def collate_fn(batch):
    max_length = max([len(data['ids']) for data in batch])
    padded_ids = []
    padded_labels = []
    for data in batch:
        ids = data['ids']
        input_ids = ids
        input_ids.extend([0]*(max_length - len(ids)))  # Padding
        labels = [-100 if x == 0 else x for x in input_ids]
        padded_ids.append(ids)
        padded_labels.append(labels)
    return {'input_ids': torch.tensor(padded_ids), 'labels': torch.tensor(padded_labels)}


parse = argparse.ArgumentParser()
parse.add_argument("--scratch", action="store_false")
parse.add_argument("--name", type=str, default="chess-mamba-v2-stockfish")
parse.add_argument("--num_layers", type=int, default=16)
parse.add_argument("--hidden_size", type=int, default=512)
parse.add_argument("--state_size", type=int, default=32)
parse.add_argument("--num_attention_heads", type=int, default=8)
parse.add_argument("--batch_size", type=int, default=64)
parse.add_argument("--gradient_accumulation_steps", type=int, default=4)
parse.add_argument("--num_epochs", type=int, default=1)
parse.add_argument("--lr", type=float, default=1e-3)
parse.add_argument("--tokenizer", type=str, default="normal")



args = parse.parse_args()

# Load the dataset
file_name="dataset_stockfish_dataset_blocks.zip"
if args.tokenizer == "normal":
    tokenizer = ChessTokenizer()
    dataset = load_from_disk("datasets/"+"normal_train_"+file_name)
    test_dataset = load_from_disk("datasets/"+"normal_test_"+file_name)
elif args.tokenizer == "simple":
    tokenizer = ChessSimpleTokenizer()
    dataset = load_from_disk("datasets/"+"simple_train_"+file_name)
    test_dataset = load_from_disk("datasets/"+"simple_test_"+file_name)
else:
    print("Invalid tokenizer")
    exit()
    

test_dataset = test_dataset.select(range(1000))
    
scratch = args.scratch
name = args.name
wandb_project = "chess-mamba"
wandb_run_name = name

print(scratch)
if scratch:
    if "mamba" in name:
        config = ChessMambaConfig(num_hidden_layers=args.num_layers,hidden_size=args.hidden_size,intermediate_size=args.hidden_size*2,state_size=args.state_size)
        model = ChessMambaModel(config,tokenizer).model
    elif "llama" in name:
        config = ChessLlamaConfig(num_hidden_layers=args.num_layers,hidden_size=args.hidden_size,n_head=args.num_attention_heads)
        model = ChessLlamaModel(config,tokenizer).model
    else:
        print("Invalid model name")
        exit()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([torch.numel(p) for p in model_parameters])
    resume_from_checkpoint = False
    print("Number of parameters: ", num_params)
else:
    if "mamba" in name:    
        model = ChessMambaModel(name,tokenizer).model
    elif "llama" in name:
        model = ChessLlamaModel(name,tokenizer).model
    else:
        print("Invalid model name")
        exit()
    name = name+"_continuation"
    resume_from_checkpoint = True
    

batch_size = args.batch_size
gradient_accumulation_steps = args.gradient_accumulation_steps
num_epochs = args.num_epochs

num_steps = len(dataset)//batch_size*num_epochs
warmup_iters = num_steps//20
max_iter = 100000

learning_rate = args.lr

os.makedirs("models/"+name, exist_ok=True)
model.config.use_cache = False

training_args = TrainingArguments(
    output_dir="models/"+name,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    gradient_accumulation_steps=gradient_accumulation_steps,
    evaluation_strategy="steps",
    eval_steps=0.05,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    remove_unused_columns=False,
    learning_rate=learning_rate,
    logging_strategy="steps",
    logging_steps=0.0005,
    report_to="wandb",
    save_strategy="steps",
    save_steps=0.05,
    resume_from_checkpoint=resume_from_checkpoint,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=test_dataset,
    data_collator=collate_fn)

wandb.init(project=wandb_project, name=wandb_run_name)

trainer.train()
model.config.tokenizer = args.tokenizer
model.save_pretrained(name,cache_dir="models")