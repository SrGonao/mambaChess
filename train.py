from tokenizer import CheessTokenizer
from model import ChessMambaModel, ChessMambaConfig, ChessLlamaModel, ChessLlamaConfig
from datasets import load_from_disk
import torch
from tqdm import tqdm
import wandb
import os
from torch.optim import AdamW
from transformers import Trainer, TrainingArguments,LlamaConfig

#Train params
#if torch.cuda.is_available():
#    device = torch.device("cuda:7")
scratch = True
name = "chess-mamba-v1.2-stockfish"
wandb_project = "chess-mamba"
wandb_run_name = name


def collate_fn(batch):
    max_length = max([len(data['ids'][0]) for data in batch])
    padded_ids = []
    padded_labels = []
    for data in batch:
        ids = data['ids'][0]
        input_ids = ids
        input_ids.extend([0]*(max_length - len(ids)))  # Padding
        labels = [-100 if x == 0 else x for x in input_ids]
        padded_ids.append(ids)
        padded_labels.append(labels)
    return {'input_ids': torch.tensor(padded_ids), 'labels': torch.tensor(padded_labels)}





# Load the dataset
file_name="dataset_stockfish_dataset_blocks.zip"
tokenizer = CheessTokenizer()

dataset = load_from_disk("train_"+file_name)
test_dataset = load_from_disk("test_"+file_name)
test_dataset = test_dataset.select(range(1000))
    

#if scratch:
model = ChessMambaModel(ChessMambaConfig())
#else:
#    model = ChessMambaModel.from_pretrained(name)
#model = ChessLlamaModel(ChessLlamaConfig())

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
num_params = sum([torch.numel(p) for p in model_parameters])
print("Number of parameters: ", num_params)
# print(LlamaConfig())
# print(ChessLlamaConfig())



batch_size = 16
gradient_accumulation_steps = 16
num_epochs = 2
num_steps = len(dataset)//batch_size*num_epochs
warmup_iters = num_steps//20
max_iter = 100000



learning_rate = 1e-3

training_args = TrainingArguments(
    name,
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
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=test_dataset,
    data_collator=collate_fn)

wandb.init(project=wandb_project, name=wandb_run_name)


trainer.train()

os.makedirs("models/"+name, exist_ok=True)

model.save_pretrained(name,cache_dir="./models")