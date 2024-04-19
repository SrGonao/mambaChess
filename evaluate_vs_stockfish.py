import pandas as pd
import argparse
from model import ChessMambaModel, ChessLlamaModel
from players import MambaPlayer,LLamaPlayer,StockfishPlayer
from tokenizer import CheessTokenizer
from game import Game
import numpy as np
import logging
import time

#logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO)


parse = argparse.ArgumentParser()
parse.add_argument("--model", type=str, default="models/chess-mamba-v1.3-stockfish")
parse.add_argument("--level", type=int, default=0)
parse.add_argument("--device", type=str, default="0")
parse.add_argument("--games", type=int, default=10)    


args = parse.parse_args()

if "mamba" in args.model: 
    model = ChessMambaModel(args.model)
    mamba = MambaPlayer(model,statistics=True)
    
elif "llama" in args.model:
    model = ChessLlamaModel(args.model)
    llama = LLamaPlayer(model,statistics=True)

else:
    print("Invalid model name")
    exit()

model.eval()
model.to(f"cuda:{args.device}")

level=args.level
stockfish = StockfishPlayer("/mnt/ssd-1/gpaulo/mambaChess/stockfish/stockfish",skill=level)

all_mistakes = []
all_best_move = []
all_results = []
all_mamba = []
all_turns = []
for j in range(args.games):
    
    #logger.info(f"Game {j+1}")
    print(f"Game {j+1}",flush=True)
    start_time = time.time()
    turn_counter = 1
    random = np.random.rand()
    if random < 0.5:
        players = [mamba,stockfish]
    else:
        players = [stockfish,mamba]
    
    game = Game(players)
    game.play(timing=True)
    all_mistakes.append(game.mistakes)
    all_best_move.append(game.best_moves)
    all_turns.append(game.turn_counter)
    result = game.result()
    if players[0] == mamba:
        print("Mamba is white",flush=True)
        all_results.append(result)
        all_mamba.append(1)
    else:
        print("Mamba is black",flush=True)
        all_results.append(1-result)
        all_mamba.append(0)
    
    end_time = time.time()
    #logger.info(f"Game time: {end_time-start_time} Player 1 time: {game.times[0]} Player 2 time: {game.times[1]}")
    print(f"Game time: {end_time-start_time} Player 1 time: {game.times[0]} Player 2 time: {game.times[1]}",flush=True)
    
results = pd.DataFrame(columns=["Result","Mistakes","Best Move","Turns","Mamba"])
for i in range(len(all_results)):
    results.loc[i]={"Result":all_results[i],"Mistakes":all_mistakes[i],"Best Move":all_best_move[i],"Turns":all_turns[i],"Mamba":all_mamba[i]}

name = args.model
split = name.split("-")
version = split[-2]+split[-1]

results.to_csv(f"results/{version}.csv",index_label="Game")
for player in players:
        try:
            player.engine.quit()
        except:
            pass
        try:
            player.kill_stockfish()
        except:
            pass
#print("Game string:",game_string)

