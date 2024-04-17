from transformers import MambaConfig, MambaForCausalLM, LlamaForCausalLM,LlamaConfig
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
        
    

class ChessLlamaModel(LlamaForCausalLM):
    config_class = ChessLlamaConfig
    base_model_prefix = "llama"
    tokenizer = CheessTokenizer()



class ChessMambaModel(MambaForCausalLM):
    config_class = ChessMambaConfig
    base_model_prefix = "mamba"
    tokenizer = CheessTokenizer()

    def get_move(self, state, board, player):
        if player == 0:
            input = self.tokenizer.encode(".")
        else:
            input = self.tokenizer.encode(" ")
        input = torch.tensor(input).unsqueeze(0)
        move=""
        original_state = deepcopy(state)
        #print(original_state.ssm_states[0][0][0][0:5])
        trial_board = deepcopy(board)
        valid = False
        input = input.to(self.device)
        #print(input)
        for i in range(6):
            output = self(input, cache_params=state, use_cache=True)
            state = output.cache_params
            logits = output.logits
            index = self.sample(logits,top_k=10)
            input = index
            part_move = self.tokenizer.decode_san(index)
            move+=part_move
            #print(move)
            try:
                trial_board.push_san(move)
                #print(valid)
                valid = True
                state = self(input, cache_params=state, use_cache=True).cache_params
                break
            except:
                continue
        if valid:
            return move,state
        else:
            return None,original_state

    def get_batched_move(self, states, boards, players):
        inputs = []
        for p in players:
            if p == 0:
                input = self.tokenizer.encode(".")
            else:
                input = self.tokenizer.encode(" ")
            inputs.append(input)
        inputs = torch.tensor(inputs)
        inputs = inputs.to(self.device)
        
        moves=["" for i in range(len(players))]
        valids = [False for i in range(len(players))]
        states = deepcopy(states)
        trial_board = deepcopy(boards)
        
        for i in range(6):
            output = self(inputs, cache_params=states, use_cache=True)
            states = output.cache_params
            logits = output.logits
            index = self.sample(logits,top_k=1)
            inputs = index
            part_move = self.tokenizer.decode_san_batch(index)
            for j,pm in enumerate(part_move):
                if valids[j] == False:
                    moves[j]+=pm
                    try:
                        trial_board[j].push_san(moves[j])
                        valids[j] = True
                    except:
                        continue
        for j in range(len(players)):
            if not valids[j]:
                moves[j] = None
        return moves,states
       

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

    def update_state(self, state, move):
        input = torch.tensor(self.tokenizer.encode(move))
        input = input.unsqueeze(0)
        input = input.to(self.device)
        for i in range(input.shape[1]):
            output = self(input[:,i].unsqueeze(0), cache_params=state, use_cache=True)
            state = output.cache_params
        return state
    
    def update_state_batch(self, states, moves):
        inputs = []
        for m in moves:
            input = torch.tensor(self.tokenizer.encode(m))
            inputs.append(input)
        inputs = torch.stack(inputs)
        inputs = inputs.to(self.device)
        for i in range(inputs.shape[1]):
            output = self(inputs[:,i], cache_params=states, use_cache=True)
            states = output.cache_params

        return states
