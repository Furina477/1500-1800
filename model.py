import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import spacy
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pickle

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.scale = math.sqrt(d_model)
    
    def forward(self, x):
        '''
        Args:
            x:(batch_size, seq_len)
        Returns:
            (batch_size, seq_len, d_model)
        '''
        return self.embedding(x) * self.scale

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=None):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0)) / d_model
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        '''
        Args:
            x:(batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model)
        '''
        return x + self.pe[:, :x.size(1), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, x):
        '''
        Args:
            (batch_size, nhead, d_model)
        Returns:
            x: (batch_size, nhead, seq_len, d_k)
        '''
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.nhead, self.d_k)
        return x.transpose(1, 2)
    
    def combine_heads(self, x):
        '''
        Args:
            x: (batch_size, nhead, seq_len, d_k)
        Returns:
            (batch_size, seq_len, d_model)
        '''
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1,2).contiguous()

        return x.view(batch_size, seq_len, self.d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        if mask is not None:
            # mask: 1表示可见，0表示被mask掉
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        # 处理所有位置都被mask导致的NaN
        attention_weights = torch.where(torch.isnan(attention_weights), torch.zeros_like(attention_weights), attention_weights)
        attention_weights = self.dropout(attention_weights)

        attention_output = torch.matmul(attention_weights, V)
        return attention_output, attention_weights

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attention_output, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask
        )

        attention_output = self.combine_heads(attention_output)

        output = self.W_o(attention_output)

        return output, attention_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x,x,x,mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        attn_output, _ = self.self_attn(x,x,x,tgt_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = x + self.dropout2(attn_output)
        x = self.norm2(x)

        ffn_output = self.ffn(x)
        x = x + self.dropout3(ffn_output)
        x = self.norm3(x)
        return x
 
class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        num_encoder_layers,
        num_decoder_layers,
        d_model,
        nhead,
        d_ff,
        dropout=0.1
    ):
        super().__init__()
        self.src_embedding = InputEmbedding(src_vocab_size, d_model)
        self.tgt_embedding = InputEmbedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEmbedding(d_model, dropout=dropout)

        self.encoder = Encoder(num_encoder_layers, d_model, nhead, d_ff, dropout)
        self.decoder = Decoder(num_decoder_layers, d_model, nhead, d_ff, dropout)

        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
    
    def generate_src_mask(self, src):
        # 1表示真实token，0表示padding
        src_mask = (src != 0).float()  # (batch_size, src_seq_len)
        return src_mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, src_seq_len)
    def generate_tgt_mask(self, tgt):
        # 生成目标句子的因果mask + padding mask
        batch_size, tgt_seq_len = tgt.size()
        
        # 因果mask：防止注意力看到未来的位置
        # torch.tril生成下三角（包括对角线）= 1，上三角 = 0
        # 结果：position i只能看到position 0..i，不能看到i+1..end
        causal_mask = torch.tril(torch.ones((tgt_seq_len, tgt_seq_len))).to(tgt.device)  # (tgt_seq_len, tgt_seq_len)
        
        # Padding mask：防止注意力看到padding positions
        # 1表示真实token，0表示padding
        pad_mask = (tgt != 0)  # (batch_size, tgt_seq_len)
        pad_mask = pad_mask.unsqueeze(1)  # (batch_size, 1, tgt_seq_len)
        
        # 合并mask：使用*进行广播相乘
        # causal_mask: (tgt_seq_len, tgt_seq_len)
        # pad_mask: (batch_size, 1, tgt_seq_len)
        # 结果: (batch_size, tgt_seq_len, tgt_seq_len)
        # tgt_mask[b, i, j] = causal_mask[i, j] * pad_mask[b, 0, j]
        tgt_mask = causal_mask * pad_mask  # 广播相乘
        
        # 转换为(batch_size, 1, tgt_seq_len, tgt_seq_len)用于multihead attention
        return tgt_mask.unsqueeze(1)
    def forward(self, src, tgt):
        src_mask = self.generate_src_mask(src)
        tgt_mask = self.generate_tgt_mask(tgt)

        src_emb = self.positional_encoding(self.src_embedding(src))
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt))

        enc_output = self.encoder(src_emb, src_mask)
        dec_output = self.decoder(tgt_emb, enc_output, src_mask, tgt_mask)

        output = self.output_layer(dec_output)

        return output