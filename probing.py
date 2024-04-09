from dataclasses import dataclass, field
from datasets import load_from_disk
from copy import deepcopy
import chess
import torch
from concept_erasure import LeaceEraser
from torch import Tensor
from torch.nn.functional import (
    binary_cross_entropy_with_logits as bce_with_logits,
)
from torch.nn.functional import (
    cross_entropy,
)
from model import ChessMambaModel
from tokenizer import CheessTokenizer
from tqdm import tqdm

class Classifier(torch.nn.Module):
    """Linear classifier trained with supervised learning."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        eraser: LeaceEraser | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.linear = torch.nn.Linear(
            input_dim, num_classes if num_classes > 2 else 1, device=device, dtype=dtype
        )
        self.linear.bias.data.zero_()
        self.linear.weight.data.zero_()
        self.eraser = eraser

    def forward(self, x: Tensor) -> Tensor:
        if self.eraser is not None:
            x = self.eraser(x)
        return self.linear(x).squeeze(-1)

    @torch.enable_grad()
    def fit(
        self,
        x: Tensor,
        y: Tensor,
        *,
        l2_penalty: float = 0.001,
        max_iter: int = 10_000,
    ) -> float:
        """Fits the model to the input data using L-BFGS with L2 regularization.

        Args:
            x: Input tensor of shape (N, D), where N is the number of samples and D is
                the input dimension.
            y: Target tensor of shape (N,) for binary classification or (N, C) for
                multiclass classification, where C is the number of classes.
            l2_penalty: L2 regularization strength.
            max_iter: Maximum number of iterations for the L-BFGS optimizer.

        Returns:
            Final value of the loss function after optimization.
        """
        optimizer = torch.optim.LBFGS(
            self.parameters(),
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
        )

        num_classes = self.linear.out_features
        loss_fn = bce_with_logits if num_classes == 1 else cross_entropy
        loss = torch.inf
        y = y.to(
            torch.get_default_dtype() if num_classes == 1 else torch.long,
        )

        def closure():
            nonlocal loss
            optimizer.zero_grad()

            # Calculate the loss function
            logits = self(x).squeeze(-1)
            loss = loss_fn(logits, y)
            if l2_penalty:
                reg_loss = loss + l2_penalty * self.linear.weight.square().sum()
            else:
                reg_loss = loss

            reg_loss.backward()
            return float(reg_loss)

        optimizer.step(closure)
        return float(loss)

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
        #return board
    #except:
        #print("Invalid move.")
        #return board

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

file_name="dataset_lichess_200k_elo_bins.zip"
test_dataset = load_from_disk("test_"+file_name)
board_states = []
internal_states = []
model = ChessMambaModel.from_pretrained("chess-mamba-v0")
tokenizer = CheessTokenizer()
model.eval()
maximum_iterations = 1000
with torch.no_grad():
    with tqdm(total=maximum_iterations) as pbar:
    
        for index,game in enumerate(test_dataset):
            board = chess.Board()   
            moves = separate_moves(game["ids"])
            state = None
            with tqdm(total=len(moves)) as pbar2:
                for m in moves:
                    for i in m:
                        state = model(torch.tensor([i]).unsqueeze(0),cache_params=state,use_cache=True).cache_params
                    move = tokenizer.decode_san(torch.tensor(m))
                    #move = "".join([tokenizer.id_to_token[i] for i in m])
                    update_board(board,move)
                    board_states.append(board.copy().fen())
                    internal_states.append(deepcopy(state))
                    pbar2.update(1)
            if index>maximum_iterations:
                break
            pbar.update(1)

# board_states = torch.stack(board_states)
# internal_states = torch.stack(internal_states)
# board_states.save("board_states.pt")
# internal_states.save("internal_states.pt")
converted_states = []
for i in internal_states:
    converted_states.append(i.ssm_states[8])
converted_states = torch.stack(converted_states)
torch.save(converted_states,"converted_states.pt")
torch.save(board_states,"board_states.pt")