import chess
import chess.engine 
from model import ChessMambaModel
from tokenizer import CheessTokenizer
import torch
from copy import deepcopy
import time
import numpy as np

class StockfishPlayer:
    def __init__(self,stockfish_location,skill=0):
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_location)
        self.skill = skill
    def get_move(self,boards,player,batch):
        self.engine.configure({"Skill Level": self.skill})
        sans = [None for i in range(len(boards))]
        for k,board in enumerate(boards):
            if board is not None:
                result = self.engine.play(board,limit=chess.engine.Limit(time=0.01))
                san = board.san(result.move)
                move_uci=board.parse_san(san)
                board.push(move_uci)
                sans[k] = san
        return sans
    

class HumanPlayer:
    def __init__(self):
        pass
    def get_move(self,board,player,batch):
        move = None
        board=board[0]
        while move is None:
            try:
                move = input("Enter move:")
                print(move)
                if move == "board":
                    print(board)
                    move = None
                    continue
                
                san_move = board.parse_san(move)
                board.push(san_move)
            except:
               print("Invalid move.")
            
               move = None
        return [move]

class MambaPlayer:
    def __init__(self,model,tokenizer,statistics=False):
        self.model = model
        self.tokenizer = tokenizer
        self.state = None
        self.game_string = ""
        if statistics:
            self.mistakes = []
            self.best_moves = []
        self.statistics = statistics
        self.turn = 1
    
    def get_move(self,board,player,batch):
        board = board[0]
        player = player[0]
        max_moves = 10
        counter = 0
        player_move = None
        self.game_string+= " " if player==1 else "."
        state = self.state
        while player_move is None:
            player_move,state = self.model.get_move(state,board,player)
            #print(counter)
            if counter>max_moves:
                legal_moves = list(board.legal_moves)[0]
                player_move = board.san(legal_moves)
                if player==0:
                    move = "." + str(player_move)
                else:
                    move = " " + str(player_move)
                model.update_state(state,move)
                if self.statistics:
                    self.mistakes.append(self.turn)
                self.turn+=1
                #player_move = legal_move
                print("Move not found.")
            counter+=1

        if self.statistics:
            stockfish.engine.configure({"Skill Level": 20})
            result = stockfish.engine.play(board,limit=chess.engine.Limit(time=0.01))
            stockfish_move = board.san(result.move)
            if stockfish_move==player_move:
                self.best_moves.append(player_move)
                #print("Best move.")
        self.state = state
        self.game_string+=player_move
        self.turn+=1
        board.push_san(player_move)
        #print(player_move)
        return [player_move]
    
    def update_state_string(self,game_string):
        game_string = game_string[-1]
        diff = len(game_string)-len(self.game_string)
        if diff>0:
            move = game_string[-diff:]
            self.game_string = game_string
            self.state = model.update_state(self.state,move)
        
        
    def verify_integrity(self):
        state_after_player_1 = self.state.ssm_states[0][0][0][0:5]
        total_game = self.game_string
        print("Total game:",total_game)
        input = torch.tensor(tokenizer.encode(total_game)).unsqueeze(0).to(model.device)
        state_model = model(input,use_cache=True).cache_params.ssm_states[0][0][0][0:5]
        print("State after player 1:",state_after_player_1,"State model:",state_model)
        assert torch.allclose(state_after_player_1,state_model)
        #print("Integrity verified.")
    
    def get_statistics(self):
        return self.mistakes,self.best_moves
    
    def reset(self):
        self.state = None
        self.game_string = ""
        self.turn = 1
        if self.statistics:
            self.mistakes = []
            self.best_moves = []
    
    def conclude_game(self,k):
        pass

class BatchMambaPlayer:
    def __init__(self,model,tokenizer,statistics=False,batch_size=1):
        self.model = model
        self.tokenizer = tokenizer
        self.states = None 
        self.game_strings = ["" for i in range(batch_size)]
        if statistics:
            self.mistakes = [[] for i in range(batch_size)]
            self.best_moves = [[] for i in range(batch_size)]
        self.statistics = statistics
        self.turn = 1
        self.batch_size = batch_size

    def get_move(self,boards,players,state=None):
        max_moves = 10
        counter = 0
        player_moves = [None for i in range(self.batch_size )]
        self.game_strings+= " " if player==1 else "."
        special_moves = []

        while None in player_moves:
            player_moves,states = self.model.get_batched_move(self.states,boards,players)
            #print("Player moves:",player_moves)
            if counter>max_moves:
                for b in range(self.batch_size):
                    if player_moves[b] is None:
                        legal_moves = list(boards[b].legal_moves)[0]
                        legal_move = boards[b].san(legal_moves)
                        player_moves[b] = " " + str(legal_move)
                        #model.update_state(states,player_moves[b])
                        special_moves.append(b)
                        if self.statistics:
                            self.mistakes[b].append(self.turn)

            counter+=1
        if self.statistics:
            stockfish.engine.configure({"Skill Level": 20})
            for b in range(self.batch_size):
                    result = stockfish.engine.play(boards[b],limit=chess.engine.Limit(time=0.01))
                    stockfish_move = boards[b].san(result.move)
                    if stockfish_move==player_moves[b]:
                        self.best_moves[b].append(self.turn)
                    #print("Best move.")
        if len(special_moves)>0:
            states = self.update_state_special(special_moves,states,self.states,player_moves)
        
        self.states = states
        for b in range(self.batch_size):
                self.game_strings[b]+=player_moves[b]
                move_uci=boards[b].parse_san(player_moves[b])
                boards[b].push(move_uci)
        self.turn+=1
        
        return player_moves

    def update_state_string(self,game_strings):
        moves = []
        for b in range(self.batch_size):
            diff = len(game_strings[b])-len(self.game_strings[b])
            if diff>0:
                move = game_strings[0][-diff:]
                self.game_strings[b] = game_strings[b]
                moves.append(move)
        self.states = model.update_state_batch(self.states,moves)
        
    def update_state_special(self,special_moves,states,old_states,player_moves):
        
        new_states = model.update_state_batch(old_states,player_moves)

        new_states_ssm = new_states.ssm_states
        new_states_conv = new_states.conv_states
        states_ssm = states.ssm_states
        states_conv = states.conv_states
        n_layers = len(new_states_ssm)
        for layer in range(n_layers):
            new_states_ssm_layer = new_states_ssm[layer]
            states_ssm_layer = states_ssm[layer]
            states_conv_layer = states_conv[layer]
            new_states_conv_layer = new_states_conv[layer]
            for b in special_moves:
                states_ssm_layer[b,:] = new_states_ssm_layer[b,:]
                states_conv_layer[b,:] = new_states_conv_layer[b,:]
        return states

    def conclude_game(self,idx):
        self.game_strings = [self.game_strings[i] for i in range(self.batch_size) if i not in idx]
        states_ssm = self.states.ssm_states
        states_conv = self.states.conv_states
        for layer in range(len(states_ssm)):
            states_ssm[layer] = states_ssm[layer][[i for i in range(self.batch_size) if i not in idx],:]
            states_conv[layer] = states_conv[layer][[i for i in range(self.batch_size) if i not in idx],:]
        self.states.ssm_states = states_ssm
        self.states.conv_states = states_conv

    def reset(self):
        self.states = None
        self.game_strings = ["" for i in range(self.batch_size)]
        self.turn = 1
        if self.statistics:
            self.mistakes = [[] for i in range(self.batch_size)]
            self.best_moves = [[] for i in range(self.batch_size)]
        
model = ChessMambaModel.from_pretrained("chess-mamba-v2-stockfish")
model.eval()
model.to("cuda:0")
tokenizer = CheessTokenizer()
turn = 1
finished = False
total_moves = 0
verify = False
batch=False
human = HumanPlayer()
if batch:
    batch_size = 1
    mamba = BatchMambaPlayer(model,tokenizer,statistics=True,batch_size=batch_size)
    states = None
    exists_mamba_player = True
else:
    mamba = MambaPlayer(model,tokenizer,statistics=True)
    batch_size = 1
    states = None
    exists_mamba_player = True

level=0
stockfish = StockfishPlayer("/mnt/ssd-1/gpaulo/mambaChess/stockfish/stockfish",skill=level)

all_mistakes = []
all_best_moves = []
all_results = []
all_boards = []

for j in range(10):
    print("Games",j+1)
    boards = [chess.Board() for i in range(batch_size)]
    start_time = time.time()
    
    turn_counter = 1
    if j < 5:
        players = [mamba,mamba]
        list_players = [0 for i in range(batch_size)]
    
    else:
        players = [mamba,mamba]
        list_players = [1 for i in range(batch_size)]
    game_strings = ["1" for i in range(batch_size)]
    
    with torch.no_grad():
        if exists_mamba_player:
            mamba.update_state_string(game_strings)
        time_p1 = 0
        time_p2 = 0
        while len(boards) >0:
            for i,player in enumerate(players):

                print(f"Player {i+1} moving.")
                start_p = time.time()
                moves = player.get_move(boards,list_players,batch_size)
                to_clean = []
                for k in range(len(boards)):
                    if i==0:
                        game_strings[k]+="."+moves[k]
                    else:
                        game_strings[k]+=" "+moves[k]+" "+str(turn_counter+1)

                    if boards[k].is_game_over():
                        to_clean.append(k)
                        all_boards.append(boards[k])
                        #print(f"Game {k} finished.")
                
                                                
                if exists_mamba_player:
                    mamba.update_state_string(game_strings)
                    mamba.conclude_game(to_clean)
                    if verify:
                        mamba.verify_integrity()
                    

                boards = [boards[k] for k in range(len(boards)) if k not in to_clean]
                game_strings = [game_strings[k] for k in range(len(game_strings)) if k not in to_clean]
                
                end_p = time.time()
                if i==0:
                    time_p1+=end_p-start_p
                else:
                    time_p2+=end_p-start_p
                #for k in range(len(boards)):
                    #print(game_strings[k])
                if len(boards)==0:
                    break
            turn_counter+=1
            #print("Turn",turn_counter)
        if exists_mamba_player:
            mistakes,best_moves = mamba.get_statistics()
            all_mistakes.append(mistakes)
            all_best_moves.append(best_moves)
            mamba.reset()
            
        #Get board result
        for board in all_boards:
            result = board.result()
            if result == "1-0":
                if j < 500:
                    all_results.append(1)
                else:
                    all_results.append(0)
                print("White wins.")
            elif result == "0-1":
                if j > 500:
                    all_results.append(1)
                else:
                    all_results.append(0)
                print("Black wins.")
            else:
                all_results.append(0.5)
                print("Draw.")
        finished = False
    end_time = time.time()
    print("Game time:",end_time-start_time,"Player 1 time:",time_p1,"Player 2 time:",time_p2)
np.savetxt(f"mistakes_stockfish{level}.txt",all_mistakes)
np.savetxt(f"best_moves_stockfish{level}.txt",all_best_moves)
np.savetxt(f"results_stockfish{level}.txt",all_results)


for player in players:
    try:
        player.engine.quit()
    except:
        pass
#print("Game string:",game_string)

