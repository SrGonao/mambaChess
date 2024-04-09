from tokenizer import CheessTokenizer
from model import ChessMambaModel, ChessMambaConfig
from datasets import load_from_disk
import torch
from tqdm import tqdm
import numpy as np
import wandb
import math
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data.distributed import DistributedSampler


def get_lr(
    iter_num: int,
    warmup_iters: int,
    learning_rate: float,
    lr_decay_iters: int,
    min_lr: float,
):
    # 1) linear warmup for warmup_iters steps
    if iter_num < warmup_iters:
        return learning_rate * iter_num / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if iter_num > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


#Train params
if torch.cuda.is_available():
    device = torch.device("cuda:1")
scratch = True
name = "chess-mamba-v0"
wandb_project = "chess-mamba"
wandb_run_name = name
min_lr = 1e-7
learning_rate = 1e-5
warmup_iters = 1000
eval_every = 50
save_every = 2000
ddp = True
maximum_iterations = 20000
gradient_accumulation_steps = 16
batch_size = 64

grad_clip=1
# Load the dataset
file_name="dataset_lichess_200k_elo_bins.zip"
tokenizer = CheessTokenizer()

dataset = load_from_disk("train_"+file_name)
test_dataset = load_from_disk("test_"+file_name)

def collate_fn(batch):
    max_length = max([len(data['ids'][0]) for data in batch])
    padded_ids = []
    for data in batch:
        ids = data['ids'][0]
        ids.extend([0]*(max_length - len(ids)))  # Padding
        padded_ids.append(ids)
       
    return {'ids': torch.tensor(padded_ids)}


if scratch:
    model = ChessMambaModel(ChessMambaConfig())
else:
    model = ChessMambaModel.from_pretrained(name)



if ddp:
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
sampler = DistributedSampler(dataset,num_replicas=ddp_world_size,rank=ddp_rank) if ddp else None
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, sampler=sampler, shuffle=False)

model.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

if ddp:
    prefix = "_orig_mod." if compile else ""
    model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
    model = DDP(model, device_ids=[ddp_local_rank])

if master_process:
    wandb.init(project=wandb_project, name=wandb_run_name)
    os.makedirs("models/"+name, exist_ok=True)

# Training loop
count=0
raw_model = model.module
log_spaces = [2,4,8,16,32,64,128,256,512]
epoch = 0
while count<maximum_iterations:
    sampler.set_epoch(epoch)
    with tqdm(total=maximum_iterations) as pbar:
        for example in train_dataloader:
            ids = example["ids"].to(device)

            lr = get_lr(count, warmup_iters, learning_rate, maximum_iterations, min_lr)
            
            if count % eval_every == 0 or count in log_spaces:
                model.eval()
                total_loss = 0
                for j,eval_example in enumerate(test_dataset):
                    eval_ids = torch.tensor(eval_example["ids"]).to(device)
                    masked_ids = eval_ids.clone()
                    masked_ids[masked_ids == 0] = -100
                
                    loss = model(eval_ids,labels=masked_ids).loss
                    total_loss+=loss.item()
                    if j>100:
                        break
                total_loss = total_loss/100
                
                model.train()
                if master_process:
                    print(f"Loss: {total_loss}")
                    wandb.log({"loss":total_loss,"lr":lr,"step":count})
            if master_process:
                if count % save_every == 0:
                    raw_model.save_pretrained("models/"+name+"/"+str(count))
                
                if count in log_spaces:
                    raw_model.save_pretrained("models/"+name+"/"+str(count))

        
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                # check when ids = 0 and set them to -100
                masked_ids = ids.clone()
                masked_ids[masked_ids == 0] = -100
                output = model(ids,labels= masked_ids)
                loss = output.loss
                loss = loss 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            count+=1
            pbar.update()
            if count>maximum_iterations:
                break
    epoch+=1
if ddp:
    destroy_process_group()
if master_process:
    raw_model.save_pretrained(name,cache_dir="./models")