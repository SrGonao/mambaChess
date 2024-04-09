from transformers import MambaConfig, MambaForCausalLM
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
    
class ChessMambaModel(MambaForCausalLM):
    config_class = ChessMambaConfig
    base_model_prefix = "mamba"
    tokenizer = CheessTokenizer()

    def get_move(self, state, board, player):
        if player == 1:
            input = self.tokenizer.encode(".")
        else:
            input = self.tokenizer.encode(" ")
        input = torch.tensor(input).unsqueeze(0)
        move=""
        original_state = deepcopy(state)
        state = deepcopy(state)
        valid = False
        #print(state.ssm_states[0][0][0][0:2])
        
        for i in range(3):
            output = self(input, cache_params=state, use_cache=True)
            state = output.cache_params
        #    print(state.ssm_states[0][0][0][0:2])
            logits = output.logits
            index = self.sample(logits,top_k=5)
            input = index.unsqueeze(0)
            part_move = self.tokenizer.decode(index)
            if part_move[0]=="r":
                part_move = part_move[1:]
            move+=part_move
            try:
                board.push_san(move)
                valid = True
                break
            except:
                continue
        if valid:
            state = self(input,cache_params=state,use_cache=True).cache_params
            return move,state,board
        else:
            return None,original_state,board

    def sample(self, logits, temperature=1.0, top_k=0, top_p=0.0):
        logits = logits[0, -1, :]
        probs = torch.softmax(logits / temperature, dim=-1)
        if top_k > 0:
            _,indexes = torch.topk(probs, top_k, dim=-1)
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask[indexes] = True
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

    def update_state(self, state, move):
        input = torch.tensor(self.tokenizer.encode(move))
        input = input.unsqueeze(0)
        for i in range(input.shape[1]):
            output = self(input[:,i].unsqueeze(0), cache_params=state, use_cache=True)
            state = output.cache_params
        return state
