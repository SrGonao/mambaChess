import chess
import torch

def occupied_positions(board_states):
    move_nums = []
    ts = []
    for b in board_states:
        board = chess.Board(b)

        t = torch.ones((64,))
        idx = 0
        for c in b.split(' ')[0].replace('/', ''):
            if c.isdigit():
                t[idx:idx+int(c)] = 0.0
                idx += int(c)
            else:
                idx+=1
        ts.append(t)
        move_nums.append(2 * int(b.split(' ')[-1]) - 2 + (1 if b.split(' ')[1] == 'b' else 0))
    return (torch.stack(ts), move_nums)


def is_check(board_states):
    move_nums = []
    ts = torch.tensor([[1.] if chess.Board(b).is_check() else [0.] for b in board_states ])
    for b in board_states:
        move_nums.append(2 * int(b.split(' ')[-1]) - 2 + (1 if b.split(' ')[1] == 'b' else 0))

    return (ts, move_nums)