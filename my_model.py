import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, attn_variant, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm        = nn.LayerNorm(hid_dim)
        self.self_attention       = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, attn_variant, device)
        self.feedforward          = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout              = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len]   #if the token is padding, it will be 1, otherwise 0
        _src, _ = self.self_attention(src, src, src, src_mask)
        src     = self.self_attn_layer_norm(src + self.dropout(_src))
        #src: [batch_size, src len, hid dim]

        _src    = self.feedforward(src)
        src     = self.ff_layer_norm(src + self.dropout(_src))
        #src: [batch_size, src len, hid dim]

        return src
    
## Added attention variant to see which variant we need to work with
class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, attn_variant, device, max_length = 500):
        super().__init__()
        self.device = device
        ## Attention variant
        self.attn_variant = attn_variant
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers        = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, attn_variant, device)
                                           for _ in range(n_layers)])
        self.dropout       = nn.Dropout(dropout)
        self.scale         = torch.sqrt(torch.FloatTensor([hid_dim])).to(self.device)

    def forward(self, src, src_mask):

        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]

        batch_size = src.shape[0]
        src_len    = src.shape[1]

        pos        = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        #pos: [batch_size, src_len]

        src        = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        #src: [batch_size, src_len, hid_dim]

        for layer in self.layers:
            src = layer(src, src_mask)
        #src: [batch_size, src_len, hid_dim]

        return src

class AdditiveAttention(nn.Module):
    def __init__(self, head_dim):
        super(AdditiveAttention, self).__init__()
        
        # Linear layers for additive attention
        self.Wa = nn.Linear(head_dim, head_dim)
        self.Ua = nn.Linear(head_dim, head_dim)
        self.V = nn.Linear(head_dim, 1)

    def forward(self, query, keys):
        # Add singleton dimensions for broadcasting
        query = query.unsqueeze(3)
        keys = keys.unsqueeze(2)

        # Apply additive attention mechanism
        features = torch.tanh(self.Wa(query) + self.Ua(keys))
        
        # Calculate attention scores
        scores = self.V(features).squeeze(-1)
        
        return scores

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, attn_variant, device):
        super().__init__()
        assert hid_dim % n_heads == 0
        
        # Initialize parameters
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.attn_variant = attn_variant

        # Linear transformations for query, key, value, and output
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

        # Scale factor for attention scores
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

        # Initialize additive attention mechanism
        self.additive_attention = AdditiveAttention(self.head_dim)

    def forward(self, query, key, value, mask=None):
        # Shapes: query = [batch size, query len, hid dim], key = [batch size, key len, hid dim], value = [batch size, value len, hid dim]

        batch_size = query.shape[0]

        # Apply linear transformations to query, key, and value
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Reshape and permute for multi-head attention
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Calculate attention scores based on the selected attention variant
        if self.attn_variant == "multiplicative":
            energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        elif self.attn_variant == "general":
            energy = torch.matmul(Q, K.permute(0, 1, 3, 2))

        elif self.attn_variant == "additive":
            energy = self.additive_attention(Q, K)

        else:
            raise Exception("Incorrect value for attention variant. Must be one of the following: multiplicative, additive, general")

        # Mask attention scores if a mask is provided
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # Apply softmax to obtain attention weights
        attention = torch.softmax(energy, dim=-1)

        # Perform weighted sum using attention weights
        x = torch.matmul(attention, V)

        # Transpose and reshape to the original shape
        x = x.transpose(-1, -2)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)

        # Apply linear transformation for the final output
        x = self.fc_o(x)

        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #x = [batch size, src len, hid dim]
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, attn_variant, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm  = nn.LayerNorm(hid_dim)
        self.ff_layer_norm        = nn.LayerNorm(hid_dim)
        self.self_attention       = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, attn_variant, device)
        self.encoder_attention    = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, attn_variant, device)
        self.feedforward          = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout              = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]

        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg     = self.self_attn_layer_norm(trg + self.dropout(_trg))
        #trg = [batch_size, trg len, hid dim]

        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg             = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        #trg = [batch_size, trg len, hid dim]
        #attention = [batch_size, n heads, trg len, src len]

        _trg = self.feedforward(trg)
        trg  = self.ff_layer_norm(trg + self.dropout(_trg))
        #trg = [batch_size, trg len, hid dim]

        return trg, attention

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads,
                pf_dim, dropout, attn_variant, device, max_length = 500):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers        = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, attn_variant, device)
                                            for _ in range(n_layers)])
        self.fc_out        = nn.Linear(hid_dim, output_dim)
        self.dropout       = nn.Dropout(dropout)
        self.scale         = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]

        batch_size = trg.shape[0]
        trg_len    = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        #pos: [batch_size, trg len]

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        #trg: [batch_size, trg len, hid dim]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        #trg: [batch_size, trg len, hid dim]
        #attention: [batch_size, n heads, trg len, src len]

        output = self.fc_out(trg)
        #output = [batch_size, trg len, output_dim]

        return output, attention

class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        ## store params to make use of model easier
        self.params = {'encoder': encoder, 'decoder': decoder,
                    'src_pad_idx': src_pad_idx, 'trg_pad_idx': trg_pad_idx}

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):

        #src = [batch size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        #src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):

        #trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        #trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        #trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask
        #trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg):

        #src = [batch size, src len]
        #trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.encoder(src, src_mask)
        #enc_src = [batch size, src len, hid dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]

        return output, attention
    
    
    class AdditiveAttention(nn.Module):
        def __init__(self, head_dim):
            super(AdditiveAttention, self).__init__()
            
            # Linear layers for additive attention
            self.Wa = nn.Linear(head_dim, head_dim)
            self.Ua = nn.Linear(head_dim, head_dim)
            self.V = nn.Linear(head_dim, 1)
    
        def forward(self, query, keys):
            # Add singleton dimensions for broadcasting
            query = query.unsqueeze(3)
            keys = keys.unsqueeze(2)
    
            # Apply additive attention mechanism
            features = torch.tanh(self.Wa(query) + self.Ua(keys))
            
            # Calculate attention scores
            scores = self.V(features).squeeze(-1)
            
            return scores
    
    class MultiHeadAttentionLayer(nn.Module):
        def __init__(self, hid_dim, n_heads, dropout, attn_variant, device):
            super().__init__()
            assert hid_dim % n_heads == 0
            
            # Initialize parameters
            self.hid_dim = hid_dim
            self.n_heads = n_heads
            self.head_dim = hid_dim // n_heads
            self.attn_variant = attn_variant
    
            # Linear transformations for query, key, value, and output
            self.fc_q = nn.Linear(hid_dim, hid_dim)
            self.fc_k = nn.Linear(hid_dim, hid_dim)
            self.fc_v = nn.Linear(hid_dim, hid_dim)
            self.fc_o = nn.Linear(hid_dim, hid_dim)
    
            # Dropout layer for regularization
            self.dropout = nn.Dropout(dropout)
    
            # Scale factor for attention scores
            self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
    
            # Initialize additive attention mechanism
            self.additive_attention = AdditiveAttention(self.head_dim)
    
        def forward(self, query, key, value, mask=None):
            # Shapes: query = [batch size, query len, hid dim], key = [batch size, key len, hid dim], value = [batch size, value len, hid dim]
    
            batch_size = query.shape[0]
    
            # Apply linear transformations to query, key, and value
            Q = self.fc_q(query)
            K = self.fc_k(key)
            V = self.fc_v(value)
    
            # Reshape and permute for multi-head attention
            Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
    
            # Calculate attention scores based on the selected attention variant
            if self.attn_variant == "multiplicative":
                energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
    
            elif self.attn_variant == "general":
                energy = torch.matmul(Q, K.permute(0, 1, 3, 2))
    
            elif self.attn_variant == "additive":
                energy = self.additive_attention(Q, K)
    
            else:
                raise Exception("Incorrect value for attention variant. Must be one of the following: multiplicative, additive, general")
    
            # Mask attention scores if a mask is provided
            if mask is not None:
                energy = energy.masked_fill(mask == 0, -1e10)
    
            # Apply softmax to obtain attention weights
            attention = torch.softmax(energy, dim=-1)
    
            # Perform weighted sum using attention weights
            x = torch.matmul(attention, V)
    
            # Transpose and reshape to the original shape
            x = x.transpose(-1, -2)
            x = x.permute(0, 2, 1, 3).contiguous()
            x = x.view(batch_size, -1, self.hid_dim)
    
            # Apply linear transformation for the final output
            x = self.fc_o(x)
    
            return x, attention
        
    class PositionwiseFeedforwardLayer(nn.Module):
        def __init__(self, hid_dim, pf_dim, dropout):
            super().__init__()
            self.fc1 = nn.Linear(hid_dim, pf_dim)
            self.fc2 = nn.Linear(pf_dim, hid_dim)
            self.dropout = nn.Dropout(dropout)
    
        def forward(self, x):
            #x = [batch size, src len, hid dim]
            x = self.dropout(torch.relu(self.fc1(x)))
            x = self.fc2(x)
    
            return x
        
    class DecoderLayer(nn.Module):
        def __init__(self, hid_dim, n_heads, pf_dim, dropout, attn_variant, device):
            super().__init__()
            self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
            self.enc_attn_layer_norm  = nn.LayerNorm(hid_dim)
            self.ff_layer_norm        = nn.LayerNorm(hid_dim)
            self.self_attention       = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, attn_variant, device)
            self.encoder_attention    = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, attn_variant, device)
            self.feedforward          = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
            self.dropout              = nn.Dropout(dropout)
    
        def forward(self, trg, enc_src, trg_mask, src_mask):
    
            #trg = [batch size, trg len, hid dim]
            #enc_src = [batch size, src len, hid dim]
            #trg_mask = [batch size, 1, trg len, trg len]
            #src_mask = [batch size, 1, 1, src len]
    
            _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
            trg     = self.self_attn_layer_norm(trg + self.dropout(_trg))
            #trg = [batch_size, trg len, hid dim]
    
            _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
            trg             = self.enc_attn_layer_norm(trg + self.dropout(_trg))
            #trg = [batch_size, trg len, hid dim]
            #attention = [batch_size, n heads, trg len, src len]
    
            _trg = self.feedforward(trg)
            trg  = self.ff_layer_norm(trg + self.dropout(_trg))
            #trg = [batch_size, trg len, hid dim]
    
            return trg, attention
    
    class Decoder(nn.Module):
        def __init__(self, output_dim, hid_dim, n_layers, n_heads,
                    pf_dim, dropout, attn_variant, device, max_length = 500):
            super().__init__()
            self.device = device
            self.tok_embedding = nn.Embedding(output_dim, hid_dim)
            self.pos_embedding = nn.Embedding(max_length, hid_dim)
            self.layers        = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, attn_variant, device)
                                                for _ in range(n_layers)])
            self.fc_out        = nn.Linear(hid_dim, output_dim)
            self.dropout       = nn.Dropout(dropout)
            self.scale         = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
    
        def forward(self, trg, enc_src, trg_mask, src_mask):
    
            #trg = [batch size, trg len]
            #enc_src = [batch size, src len, hid dim]
            #trg_mask = [batch size, 1, trg len, trg len]
            #src_mask = [batch size, 1, 1, src len]
    
            batch_size = trg.shape[0]
            trg_len    = trg.shape[1]
    
            pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
            #pos: [batch_size, trg len]
    
            trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
            #trg: [batch_size, trg len, hid dim]
    
            for layer in self.layers:
                trg, attention = layer(trg, enc_src, trg_mask, src_mask)
    
            #trg: [batch_size, trg len, hid dim]
            #attention: [batch_size, n heads, trg len, src len]
    
            output = self.fc_out(trg)
            #output = [batch_size, trg len, output_dim]
    
            return output, attention
    
    class Seq2SeqTransformer(nn.Module):
        def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
            super().__init__()
            ## store params to make use of model easier
            self.params = {'encoder': encoder, 'decoder': decoder,
                        'src_pad_idx': src_pad_idx, 'trg_pad_idx': trg_pad_idx}

            self.encoder = encoder
            self.decoder = decoder
            self.src_pad_idx = src_pad_idx
            self.trg_pad_idx = trg_pad_idx
            self.device = device

        def make_src_mask(self, src):

            #src = [batch size, src len]

            src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
            #src_mask = [batch size, 1, 1, src len]

            return src_mask

        def make_trg_mask(self, trg):

            #trg = [batch size, trg len]

            trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
            #trg_pad_mask = [batch size, 1, 1, trg len]

            trg_len = trg.shape[1]

            trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
            #trg_sub_mask = [trg len, trg len]

            trg_mask = trg_pad_mask & trg_sub_mask
            #trg_mask = [batch size, 1, trg len, trg len]

            return trg_mask

        def forward(self, src, trg):

            #src = [batch size, src len]
            #trg = [batch size, trg len]

            src_mask = self.make_src_mask(src)
            trg_mask = self.make_trg_mask(trg)

            #src_mask = [batch size, 1, 1, src len]
            #trg_mask = [batch size, 1, trg len, trg len]

            enc_src = self.encoder(src, src_mask)
            #enc_src = [batch size, src len, hid dim]

            output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

            #output = [batch size, trg len, output dim]
            #attention = [batch size, n heads, trg len, src len]

            return output, attention