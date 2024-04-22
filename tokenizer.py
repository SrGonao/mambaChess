import torch
#possible chess squares
chess_squares = ["a1","a2","a3","a4","a5","a6","a7","a8",
                "b1","b2","b3","b4","b5","b6","b7","b8",
                "c1","c2","c3","c4","c5","c6","c7","c8",
                "d1","d2","d3","d4","d5","d6","d7","d8",
                "e1","e2","e3","e4","e5","e6","e7","e8",
                "f1","f2","f3","f4","f5","f6","f7","f8",
                "g1","g2","g3","g4","g5","g6","g7","g8",
                "h1","h2","h3","h4","h5","h6","h7","h8",
                "a","b","c","d","e","f","g","h",
                "r1","r2","r3","r4","r5","r6","r7","r8"]
rows = ["r1","r2","r3","r4","r5","r6","r7","r8"]
#possible chess pieces
chess_pieces = ["N","B","R","Q","K"]
#special moves
special_moves = ["O-O","O-O-O","x","=","+","#"]
#turn numbers
chess_numbers = ["0","1","2","3","4","5","6","7","8","9"]

delimiters= ["."," "]
pad_token = ["PAD"]
total_vocab = pad_token + chess_squares + chess_pieces + special_moves + chess_numbers + delimiters

class ChessTokenizer():
    def __init__(self):
        self.vocab = total_vocab
        self.vocab_size = len(self.vocab)
        self.pattern = ""
        self.token_to_id,self.id_to_token = self._build_vocab()
        
    def encode(self, text):
        encoded = []
        if text == " ":
            return [self.token_to_id[" "]]
        else:
            if text[0] == " ":
                encoded.append(self.token_to_id[" "])
                text = text[1:]
            if text[0] == ";":
                text = text[1:]
            split_text = text.split(" ")
        
            for token in split_text:
                if len(token)==0:
                    continue
                if token[0] in chess_numbers:
                    period_split = token.split(".")
                    numbers = period_split[0]
                    if len(period_split)>1:
                        move = period_split[1]
                        for number in numbers:
                            encoded.append(self.token_to_id[number])
                        encoded.append(self.token_to_id["."])
                        tokens = self.split_move(move)
                        if tokens == "stop":
                            encoded=[""]
                        for t in tokens:
                            encoded.append(self.token_to_id[t])
                    else:
                        for number in numbers:
                            encoded.append(self.token_to_id[number])
                else:
                    tokens = self.split_move(token)
                    if tokens == "stop":
                        encoded=[""]
                    for t in tokens:
                        try:
                            encoded.append(self.token_to_id[t])
                        except:
                            print("Error in token: ",t)
                            print("Token list: ",tokens)
                            print("Text: ",text)
                            print("Encoded: ",encoded)
                            exit()
                encoded.append(self.token_to_id[" "])

            return encoded
    
    def encode_batch(self, texts):
        return [self.encode(text) for text in texts]

    
    def decode(self, tokens):
        if isinstance(tokens,torch.Tensor):
            tokens = tokens.tolist()
        string = ""
        for token in tokens:
            decoded = self.id_to_token[token]
            if decoded in rows:
                decoded = decoded[1:]
            string+=decoded
        return string
    
    def decode_batch(self, batch_tokens):
        return [self.decode_san(batch) for batch in batch_tokens]
            
        
    def _build_vocab(self):
        token_to_id = {token: i for i, token in enumerate(self.vocab)}
        id_to_token = {i: token for i, token in enumerate(self.vocab)}
        return token_to_id, id_to_token
    
    def split_move(self,move):
        #move will be a string. It could start either with a chess piece, a chess square or be a special move
        tokens = []
        #check if the move starts with a chess piece
        max_loops = 100
        counter = 0
        original_move = move
        while len(move)>0:
            if len(move)>0:
                check_piece = move[0]
                if check_piece in chess_pieces:
                    tokens.append(check_piece)
                    move = move[1:]
            #check if the move starts with a chess square
            if len(move)>1:
                check_square = move[0:2]
                if check_square in chess_squares:
                    tokens.append(check_square)
                    move = move[2:]
            #check if the move starts with a single letter chess square (meaning a move in a column or row)
            if len(move)>0:
                check_single_square = move[0]
                if check_single_square in chess_squares:
                    tokens.append(check_single_square)
                    move = move[1:]
                if check_single_square in chess_numbers:
                    tokens.append("r"+check_single_square)
                    move = move[1:]
            
            #check if the move starts with a special move
            if len(move)>0:
                check_special = move[0]
                if check_special == "O":
                    if len(move)>4:
                        if move[:5] == "O-O-O":
                            move = move[5:]
                            tokens.append("O-O-O")
                    else:
                        if move[:3] == "O-O":
                            move = move[3:]
                            tokens.append("O-O")
                elif check_special in special_moves:
                    tokens.append(check_special)
                    move = move[1:]
            #check if the move starts with a delimiter
            if len(move)>0:
                check_delimiter = move[0]
                if check_delimiter in delimiters:
                    tokens.append(check_delimiter)
                    move = move[1:]
            counter += 1
            if counter > max_loops:
                print("Error in move: ",original_move
                      ," with tokens: ",tokens,
                      " and move: ",move)
                return "stop"
                
            
        return tokens
simple_vocab = ["PAD","."," ","a","b","c","d","e","f","g","h","1","2","3","4","5","6","7","8","9","0","N","B","R","Q","K","q","k","x","=","+","#"]

class ChessSimpleTokenizer():
    def __init__(self):
        self.vocab = simple_vocab
        self.vocab_size = len(self.vocab)
        self.pattern = ""
        self.token_to_id,self.id_to_token = self._build_vocab()
        
    def encode(self, text):
        text = text.replace("O-O-O","k")
        text = text.replace("O-O","q")
        encoded = [self.token_to_id[s] for s in text]
        return encoded
    
    def encode_batch(self, texts):
        return [self.encode(text) for text in texts]

    def decode(self, tokens):
        if isinstance(tokens,torch.Tensor):
            tokens = tokens.tolist()
        decoded = "".join([self.id_to_token[token] for token in tokens])
        decoded = decoded.replace("k","O-O-O")
        decoded = decoded.replace("q","O-O")
        return decoded

    def decode_batch(self, tokens):
        return [self.decode(token) for token in tokens]
         
    def _build_vocab(self):
        token_to_id = {token: i for i, token in enumerate(self.vocab)}
        id_to_token = {i: token for i, token in enumerate(self.vocab)}
        return token_to_id, id_to_token
    