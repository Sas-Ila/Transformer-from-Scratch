import torch
import torch.nn as nn
import math



class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:

        super.__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super.__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(self.seq_len, self.d_model)

        # Create a vector of shape (seq_len, 1)
        pos = torch.arange(0, self.seq_len, dtype=float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))

        pe[:, 0::2] = torch.sin( pos * div_term)
        pe[:, 1::2] = torch.cos( pos * div_term)


        pe = pe.unsqueeze(0) #tensor of shape  (1, seq_len, d_model)

        self.register_buffer('pe', pe)
        

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)



class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10 ** -6) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(d_ff, d_model)

    def forward(self, x):

        x = self.lin1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        return x


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, nb_head: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.nb_head = nb_head

        assert d_model % nb_head == 0, "d_model is not divisible by nb_head"

        self.d_k = d_model//nb_head

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        self.Wo = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):

        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask:
            attention_scores.masked_fill_(mask == 0, -10**9)

        attention_scores = attention_scores.softmax(dim = -1)

        if dropout:
            attention_scores = dropout(attention_scores)
        
        return attention_scores @ value, attention_scores



    def forward(self, q, k, v, mask):

        query = self.Wq(q)
        key = self.Wk(k)
        value = self.Wv(v)

        query = query.view(query.shape[0], query.shape[1], self.nb_head, self.dk).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.nb_head, self.dk).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.nb_head, self.dk).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.nb_head * self.d_k)

        return self.Wo(x)
    

class ResidualConnections(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.LN = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.LN(x)))
    

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnections(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block(x))

        return x
    

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None: 
        super().__init__()
        self.layers = layers
        self.LN = LayerNormalization()
    
    def forward(self, x, mask):

        for layer in self.layers:
            x = layer(x, mask)
        return self.LN(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block

        self.residual_connection = nn.ModuleList([ResidualConnections(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):

        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.self_attention_block(encoder_output, encoder_output, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x
    

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.LN = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):

        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.LN(x)
    

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()

        self.proj_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj_layer(x), dim = -1)
    

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embd: InputEmbeddings, tgt_embd: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, proj_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embd = src_embd
        self.tgt_embd = tgt_embd
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj_layer = proj_layer
    
    def encode(self, src, src_mask):
        src = self.src_embd(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embd(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.proj_layer(x)




def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, nb_head: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embd = InputEmbeddings(d_model, src_vocab_size)
    tgt_embd = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, nb_head, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, nb_head, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, nb_head, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    proj_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embd, tgt_embd, src_pos, tgt_pos, proj_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer
    










