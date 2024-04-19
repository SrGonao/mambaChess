import chess
import chess.engine
import torch


class StockfishPlayer:
    def __init__(self,stockfish_location,skill=0):
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_location)
        self.skill = skill
    def get_move(self,board,player):
        self.engine.configure({"Skill Level": self.skill})
        result = self.engine.play(board,limit=chess.engine.Limit(time=0.01))
        san = board.san(result.move)
        move_uci=board.parse_san(san)
        board.push(move_uci)
        return san

    def reset(self):
        pass
    

class HumanPlayer:
    def __init__(self):
        pass
    def get_move(self,board,player):
        move = None
        board=board
        while move is None:
            try:
                move = input("Enter move:")
                #print(move)
                if move == "board":
                    print(board)
                    move = None
                    continue
                if move =="r":
                    legal_moves = list(board.legal_moves)
                    move = board.san(legal_moves[0])
                    
                san_move = board.parse_san(move)
                board.push(san_move)
            except:
               print("Invalid move.")
            
               move = None
        return move
    def reset(self):
        pass

class ModelPlayer:
    
    def __init__(self,model,statistics=False):
        self.model = model
        self.state = None
        self.game_string = ""
        if statistics:
            self.mistakes = []
            self.best_moves = []
        self.statistics = statistics
        self.turn = 1
        self.stockfish = chess.engine.SimpleEngine.popen_uci("/mnt/ssd-1/gpaulo/mambaChess/stockfish/stockfish")

    def get_move(self,board,player):
        player_move = None
        state = self.state
        player_move,state,mistake = self.model.get_move(state,board,player)
        
        self.game_string+= " " if player==1 else "."
        
        if self.statistics:
            if mistake:
                self.mistakes.append(self.turn)
            self.stockfish.configure({"Skill Level": 20})
            result = self.stockfish.play(board,limit=chess.engine.Limit(time=0.01))
            stockfish_move = board.san(result.move)
            if stockfish_move==player_move:
                self.best_moves.append(self.turn)
                #print("Best move.")
        self.state = state
        self.game_string+=player_move
        self.turn+=1
        board.push_san(player_move)
        #print(player_move)
        return player_move
    
    def update_state_string(self,game_string):
        diff = len(game_string)-len(self.game_string)
        if diff>0:
            move = game_string[-diff:]
            self.game_string = game_string
            self.state = self.model.update_state(self.state,move)
    
    def get_statistics(self):
        return self.mistakes,self.best_moves
    
    def reset(self):
        self.state = None
        self.game_string = ""
        self.turn = 1
        if self.statistics:
            self.mistakes = []
            self.best_moves = []
    
    def kill_stockfish(self):
        self.stockfish.quit()

class MambaPlayer(ModelPlayer):
    def __init__(self,model,statistics=False):
        super().__init__(model,statistics)
        self.tokenizer = model.tokenizer
        
    def verify_integrity(self):
        
        state_after_player_1 = self.state.ssm_states[0][0][0]

        length = len(state_after_player_1)
        random_indexes = torch.randint(0,length,(5,))
        state_after_player_1 = state_after_player_1[random_indexes]
        total_game = self.game_string
        #print("Total game:",total_game)
        input = torch.tensor(self.tokenizer.encode(total_game)).unsqueeze(0).to(self.model.device)

        _, state = self.model.forward(input)
        state_model = state.ssm_states[0][0][0][random_indexes]
        print("State after player 1:",state_after_player_1,"State model:",state_model)
        assert torch.allclose(state_after_player_1,state_model)
        #print("Integrity verified.")
    
    
     
class LLamaPlayer(ModelPlayer):
    def __init__(self,model,statistics=False):
        super().__init__(model,statistics)
        