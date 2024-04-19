from game import Game
from model import ChessMambaModel
from players import MambaPlayer,LLamaPlayer,HumanPlayer,StockfishPlayer
import argparse


def main():
    parser = argparse.ArgumentParser(description='Play a game of chess.')
    parser.add_argument('--mamba', type=str, help='Path to mamba model.')
    parser.add_argument('--stockfish', type=bool, help='Play against stockfish?')
    parser.add_argument('--skill', type=int, help='Skill level of stockfish.')
    args = parser.parse_args()
    players = []
    if args.mamba:
        model = ChessMambaModel(args.mamba)
        model.to("cuda:0")  
        model.eval()
        mamba_player = MambaPlayer(model,True)

        players.append(mamba_player)
    if args.stockfish:
        stockfish_location = "/mnt/ssd-1/gpaulo/mambaChess/stockfish/stockfish"
        stockfish_player = StockfishPlayer(stockfish_location,args.skill)
        players.append(stockfish_player)
    human_player = HumanPlayer()
    players.append(human_player)
    game = Game(players)
    game.play(verbose=["turn","player","move"],timing=True)
    print("Mistakes:",game.mistakes)
    print("Best moves:",game.best_moves)

if __name__ == "__main__":
    main()