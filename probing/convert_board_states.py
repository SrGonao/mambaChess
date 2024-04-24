import torch
from tqdm import tqdm
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--board-states')
    parser.add_argument('--output-file')
    args = parser.parse_args()

    board_states = torch.load(args.board_states)
    ts = []
    for b in tqdm(board_states):
        t = torch.ones((64,))
        idx = 0
        for c in b.split(' ')[0].replace('/', ''):
            if c.isdigit():
                t[idx:idx+int(c)] = 0.0
                idx += int(c)
            else:
                idx+=1
        ts.append(t)
    torch.save(torch.stack(ts), args.output_file)