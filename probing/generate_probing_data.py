from argparse import ArgumentParser
from datasets import load_from_disk
from copy import deepcopy
import chess
import torch
from model import ChessMambaModel
from tokenizer import CheessTokenizer
from tqdm import tqdm
from pathlib import Path


def update_board(board,move):
    #try:
        if "." in move:
            move = move.split(".")[1]
            move = move.strip() 
            if "r1" or "r2" or "r3" or "r4" or "r5" or "r6" or "r7" or "r8" in move:
                #substitute r for the number
                move.replace("r","")
            board.push_san(move)
        else:
            move = move.strip()
            board.push_san(move)

def separate_moves(ids):
    ids = ids[0]
    moves = []
    partial=[]
    for i in ids:
        partial.append(i)
        if i==103:
            moves.append(partial)
            partial=[]
    return moves

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dataset')
    parser.add_argument('--model')
    parser.add_argument('--output-dir')
    parser.add_argument('--max-iters', type=int, default=100)
    args = parser.parse_args()

    device = args.device
    dataset = args.dataset
    model_path = args.model
    maximum_iterations = args.max_iters
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_dataset = load_from_disk(dataset)
    board_states = []
    internal_states = []
    internal_activations = []
    model = ChessMambaModel(model_path)
    model.to(device)

    tokenizer = CheessTokenizer()
    model.eval()
    with torch.no_grad():
        with tqdm(total=maximum_iterations) as pbar:
            for index,game in enumerate(test_dataset):
                board = chess.Board()   
                moves = separate_moves(game["ids"])
                state = None
                with tqdm(total=len(moves)) as pbar2:
                    for m in moves:
                        for i in m:
                            out = model.forward(torch.tensor([i]).unsqueeze(0).to(device), state=state)[0]
                            state = out.cache_params
                            activations = out.hidden_states
                        move = tokenizer.decode_san(torch.tensor(m))
                        update_board(board,move)
                        board_states.append(board.copy().fen())
                        internal_states.append(torch.stack(list(deepcopy(state).ssm_states.values())).cpu())
                        internal_activations.append(torch.stack(activations).cpu())
                        pbar2.update(1)
                if index>maximum_iterations:
                    break
                pbar.update(1)

    torch.save(torch.stack(internal_states), output_dir / Path("states.pt"))
    torch.save(torch.stack(internal_activations), output_dir / Path("activations.pt"))
    torch.save(board_states, output_dir / Path("board_states.pt"))