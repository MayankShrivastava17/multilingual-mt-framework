import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_tokens: int, d_model: int = 512, nhead: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(num_tokens, d_model)
        self.tgt_embed = nn.Embedding(num_tokens, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.generator = nn.Linear(d_model, num_tokens)

    def encode(self, src, src_key_padding_mask=None):
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        memory = self.transformer.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return memory

    def decode(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        y = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        y = self.pos_enc(y)
        out = self.transformer.decoder(
            y, memory, tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        logits = self.generator(out)
        return logits

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        memory = self.encode(src, src_key_padding_mask)
        logits = self.decode(tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return logits