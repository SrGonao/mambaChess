import chess
from model import ChessMambaModel
from tokenizer import CheessTokenizer
import torch
from copy import deepcopy

def generate_move(model,tokenizer,player,previous_moves,state=None):
    if state is None:
        input=previous_moves
        

board = chess.Board()
model = ChessMambaModel.from_pretrained("chess-mamba-v0")
model.eval()
tokenizer = CheessTokenizer()
state=None
turn = 1
finished = False
game_string = ""
total_moves = 0
verify = False
with torch.no_grad():
    while not finished:
        if turn>1:
            str_turn = " "+str(turn)
        else:
            str_turn = "1"
        game_string+=str_turn
        
        print("Turn:", turn)
        turn_ids = torch.tensor(tokenizer.encode(str_turn)).unsqueeze(0)
        #print("Encoding state.")
        for ti in turn_ids[0]:
            state = model(ti.unsqueeze(0).unsqueeze(0),cache_params=state,use_cache=True).cache_params
        
        turn_state = state
        player_move= None
        print("Player 1 moving.")
        counter = 0
        while player_move is None:
            player_move,p1_state,board = model.get_move(turn_state,board,1)
            #print(p1_state.ssm_states[0][0][0][0:5])
            counter+=1
            total_moves+=1
            if counter>10:
                legal_moves = list(board.legal_moves)
                player_move = " " + str(legal_moves[0])
                p1_state = model.update_state(p1_state,player_move)
                print("Random P1.")
                board.push(legal_moves[0])
        
        print("Player 1 move: ",player_move)
        
        game_string+="." + player_move
        if verify:
            state_after_player_1 = p1_state.ssm_states[0][0][0][0:5]
            state_model = model(torch.tensor(tokenizer.encode(game_string)).unsqueeze(0),use_cache=True).cache_params.ssm_states[0][0][0][0:5]
            #print(state_after_player_1)
            #print(state_model)
            assert torch.allclose(state_after_player_1,state_model)
        finished = board.is_game_over()
        if finished:
            break
            #pass
        player_move = None
        print("Player 2 moving.")
        counter = 0
        while player_move is None:
            player_move,p2_state,board = model.get_move(p1_state,board,2)
            #print(state.ssm_states[0][0][0][0:5])
            counter+=1
            total_moves+=1
            if counter>10:
                legal_moves = list(board.legal_moves)
                player_move = " "+str(legal_moves[0])
                p2_state = model.update_state(p2_state,player_move)
                print("Random P2.")
                board.push(legal_moves[0])
        
        game_string+=" " + player_move
        print("Player 2 move: ",player_move)
        if verify:
            state_after_player_2 = p2_state.ssm_states[0][0][0][0:5]
            state_model = model(torch.tensor(tokenizer.encode(game_string)).unsqueeze(0),use_cache=True).cache_params.ssm_states[0][0][0][0:5]
            #print(state_after_player_2)
            #print(state_model)
            assert torch.allclose(state_after_player_2,state_model)
        state = p2_state
        finished = board.is_game_over()
        if finished:
            break
        turn+=1
