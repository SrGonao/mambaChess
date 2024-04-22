import chess
import torch
import chess.engine 

from copy import deepcopy
import time
from players import MambaPlayer,LLamaPlayer

class Game:
    def __init__(self,players):
        self.board = chess.Board()
        self.players = players
        self.game_string = "1"
        self.exists_mamba_player = False
        for player in players:
            player.reset()
                
    def play(self,verify=False,verbose=[],timing=True):
        self.reset()
        board = self.board
        
        game_string = self.game_string
        players = self.players
        turn_counter = 1
        with torch.no_grad():
            time_p1 = 0
            time_p2 = 0
            while not board.is_game_over():
                if "turn" in verbose:
                    print("Turn",turn_counter)
                for i,player in enumerate(players):
                    if isinstance(player,MambaPlayer) or isinstance(player,LLamaPlayer):
                        player.update_state_string(game_string)
                        if verify:
                            player.verify_integrity()
                    if timing:
                        start_p = time.time()
                    move = player.get_move(board,i)
                    if i==0:
                        game_string+="."+move
                    else:
                        game_string+=" "+move+" "+str(turn_counter+1)

                    if "player" in verbose:
                        print(f"Player {i+1} moving.")
                    if "move" in verbose:
                        print(move)
                    if "board" in verbose:
                        print(game_string)
                    if board.is_game_over():
                        break
                    
                    if timing:
                        end_p = time.time()
                        if i==0:
                            time_p1+=end_p-start_p
                        else:
                            time_p2+=end_p-start_p
                turn_counter+=1
        self.board = board
        self.game_string = game_string
        self.turn_counter = turn_counter
        if time:
            self.times = [time_p1,time_p2]
        
        for player in players:
            if isinstance(player,MambaPlayer) or isinstance(player,LLamaPlayer):
                mistakes,best_moves = player.get_statistics()
                self.mistakes = mistakes
                self.best_moves = best_moves
        
    def result(self,verbose=True):
        result = self.board.result()
        value = 0
        if result == "1-0":
            if verbose:
                print("White wins.")
            value = 1
        elif result == "0-1":
            if verbose:
                print("Black wins.")
            value = 0
        else:
            if verbose:
                print("Draw.")
            value = 0.5
        return value
        
            
    def reset(self):
        self.board = chess.Board()
        self.turn_counter = 1
        self.game_string = "1"
        for player in self.players:
            player.reset()

