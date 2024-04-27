from argparse import ArgumentParser
from pathlib import Path
import torch
import chess
from torch import Tensor
from torch.nn.functional import (
    binary_cross_entropy_with_logits as bce_with_logits,
    sigmoid
)
from torch.nn.functional import (
    cross_entropy,
)
from .board_conversions import *

class Classifier(torch.nn.Module):
    """Linear classifier trained with supervised learning."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.linear = torch.nn.Linear(
            input_dim, num_classes if num_classes > 2 else 1, device=device, dtype=dtype
        )
        self.linear.bias.data.zero_()
        self.linear.weight.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(self.linear(x).squeeze(-1))

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
        loss_fn = bce_with_logits# if num_classes == 1 else cross_entropy
        loss = torch.inf
        # y = y.to(
        #     torch.get_default_dtype() if num_classes == 1 else torch.long,
        # )

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

        for i in range(100):
            optimizer.step(closure)
        return float(loss)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--train-data')
    parser.add_argument('--test-data')
    args = parser.parse_args()

    device = args.device
    train_dir = Path(args.train_data)
    test_dir = Path(args.test_data)

    convert_board_states = occupied_positions

    LAYERS = [6]
    print('[')
    for layer in LAYERS:
        states = torch.load(train_dir  / Path("states.pt")).to(device)
        board_states = torch.load(train_dir / Path("board_states.pt"))
        converted_board_states = convert_board_states(board_states)[0].to(device)

        move_nums = convert_board_states(board_states)[1]
        move_nums_tensor = torch.tensor(move_nums).unsqueeze(-1).to(device)
        states = states.flatten(start_dim=2)

        #print(converted_board_states.size())

        states = states[:, layer, :]

        # Add move nums as extra information to probe
        #states = torch.cat((states, move_nums_tensor), dim=-1)

        classifier = Classifier(
            input_dim=states.shape[1],
            num_classes=converted_board_states.shape[1],
            device=device)

        classifier.fit(states, converted_board_states)

        states = torch.load(test_dir / Path("states.pt")).to(device)
        board_states = torch.load(test_dir / Path("board_states.pt"))
        converted_board_states = convert_board_states(board_states)[0].to(device)

        move_nums = convert_board_states(board_states)[1]
        move_nums_tensor = torch.tensor(move_nums).unsqueeze(-1).to(device)
        states = states.flatten(start_dim=2)

        for move_num in range(1, max(move_nums)):
            states_ = states[torch.Tensor(move_nums) == move_num, :]
            converted_board_states_ = converted_board_states[torch.Tensor(move_nums) == move_num, :]
            move_nums_tensor_ = move_nums_tensor[torch.Tensor(move_nums) == move_num, :]
            num_pieces = (torch.sum(converted_board_states_) / converted_board_states_.shape[0]).item()
            states_ = states_[:, layer, :]

            # Add move nums as extra information to probe
            #states_ = torch.cat((states_, move_nums_tensor_), dim=-1)

            preds = (classifier.forward(states_) > 0.5).type_as(converted_board_states_)
            #preds = torch.zeros_like(preds)
            preds = preds == converted_board_states_
            acc = (torch.sum(preds.to(torch.float32).flatten()) / preds.flatten().size()[0]).item()
            print('(', layer, ',', move_num, ",", num_pieces, ",", acc, '),')
    print(']')
