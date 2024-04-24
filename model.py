from transformers import MambaConfig, MambaForCausalLM, LlamaForCausalLM,LlamaConfig, AutoModelForCausalLM
from tokenizer import CheessTokenizer
import torch
from copy import deepcopy

class ChessMambaConfig(MambaConfig):
    model_type = "mamba"

    def __init__(
        self,
        vocab_size=104,
        hidden_size=512,
        state_size=32,
        num_hidden_layers=16,
        intermediate_size=1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.use_cache = False

class ChessLlamaConfig(LlamaConfig):
    model_type = "llama"

    def __init__(
        self,
        vocab_size=104,
        hidden_size=512,
        num_hidden_layers=8,
        n_head=8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = hidden_size*3
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = n_head
        self.num_key_value_heads = n_head
        self.bos_token_id = 0
        self.eos_token_id = 0
        
class ChessModel:

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def sample(self, logits, temperature=1.0, top_k=0, top_p=0.0):
        logits = logits[:, -1, :]
        probs = torch.softmax(logits / temperature, dim=-1)
        batch_size = logits.shape[0]
        if top_k > 0:
            _,indexes = torch.topk(probs, top_k, dim=-1)
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask[torch.arange(batch_size).unsqueeze(1),indexes] = True
            probs[~mask] = 0
        if top_p > 0.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            sorted_probs = sorted_probs.cumsum(dim=-1)
            sorted_indices = sorted_indices[sorted_probs > top_p]
            sorted_probs = sorted_probs[sorted_probs > top_p]
            sorted_probs /= sorted_probs.sum()
            index = torch.multinomial(sorted_probs, 1)
            index = sorted_indices[index]
        else:
            index = torch.multinomial(probs, 1)
        return index

    def get_move(self, state, board, player):
        
        if player == 0:
            input = self.tokenizer.encode(".")
        else:
            input = self.tokenizer.encode(" ")
        input = torch.tensor(input).unsqueeze(0)
        
        move=""
        
        trial_board = deepcopy(board)
        
        original_state = deepcopy(state)
        valid = False
        input = input.to(self.device)
        original_input = input
        max_moves = 5
        counter = 0
        while counter<max_moves and not valid:
            state = deepcopy(original_state)
            input = original_input
            for i in range(6):
                output,state = self.forward(input,state)
                logits = output.logits
                index = self.sample(logits,top_k=5)
                input = index
                part_move = self.tokenizer.decode_san(index)
                #print(part_move)
                move+=part_move
                #print(move)
                try:
                    trial_board.push_san(move)
                    valid = True
                    _, state = self.forward(input, state)
                    break
                except:
                    if part_move == " ":
                        break
                    else:
                        continue
            counter+=1
                    
        if valid:
            return move,state,0
        else:
            legal_moves = list(board.legal_moves)[0]
            legal_move = board.san(legal_moves)
            if player == 0:
                move = "." + str(legal_move)
                state = self.update_state(original_state,move)
            else:
                move = " " + str(legal_move)
                state = self.update_state(original_state,move)
            #print("random")
            return legal_move,state,1   
    def forward(self,inputs,state):
        pass

    def eval(self):
        self.model.eval()

    def to(self,device):
        self.device = device
        self.model.to(device)
      

class ChessLlamaModel(ChessModel):

    config_class = ChessLlamaConfig
    base_model_prefix = "llama"
    
    def __init__(self,config_or_path):
        super().__init__()
        if isinstance(config_or_path, str):
            self.model = LlamaForCausalLM.from_pretrained(config_or_path)
        else:
            self.model = LlamaForCausalLM(config_or_path)
        self.model.config.use_cache = True
        self.tokenizer = CheessTokenizer()

    def forward(self,inputs,state):
        outputs = self.model(inputs, past_key_values=state, use_cache=True)
        return outputs,outputs.past_key_values
    
    def update_state(self, state, move):
        input = torch.tensor(self.tokenizer.encode(move))
        input = input.unsqueeze(0)
        input = input.to(self.device)
        for i in range(input.shape[1]):
            output = self.model(input[:,i].unsqueeze(0), past_key_values=state, use_cache=True)
            state = output.past_key_values
        return state

class ChessMambaModel(ChessModel):

    base_model_prefix = "mamba"
 
    def __init__(self,config_or_path):
        super().__init__()
        if isinstance(config_or_path, str):
            self.model = MambaForCausalLM.from_pretrained(config_or_path)
        else:
            self.model = MambaForCausalLM(config_or_path)
        self.model.config.use_cache = True
        self.tokenizer = CheessTokenizer()


    def forward(self,inputs,state=None):
        outputs = self.model(inputs, cache_params=state, output_hidden_states=True, use_cache=True)
        return outputs,outputs.cache_params

    def update_state(self, state, move):
        input = torch.tensor(self.tokenizer.encode(move))
        input = input.unsqueeze(0)
        input = input.to(self.device)
        for i in range(input.shape[1]):
            _,state = self.forward(input[:,i].unsqueeze(0), state)
            
        return state
    