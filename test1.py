import torch
import torch.nn as nn

class selfAttention(nn.Module):
    def __init__(self, embedsize, heads):
        super(selfAttention, self).__init__()
        self.embedSize = embedsize
        self.heads = heads
        self.headDim = embedsize // heads

        assert (self.headDim * heads == embedsize), "Embed size needs to be div by heads"

        self.values = nn.Linear(self.headDim, self.headDim, bias=False)#设置wv
        self.keys = nn.Linear(self.headDim, self.headDim, bias=False)  # 设置wv
        self.queries = nn.Linear(self.headDim, self.headDim, bias=False)  # 设置wv
        self.fc_out = nn.Linear(embedsize,embedsize)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]#一次同时训练的数量
        val_len, key_len, que_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, val_len, self.heads, self.headDim)
        keys = values.reshape(N, key_len, self.heads, self.headDim)
        query = values.reshape(N, que_len, self.heads, self.headDim)#??????

        energy = torch.einsum("nqhd,nkhd->nhqk",[query, keys])
        #energy shape :(N, heads, querLen, keyLen)
        #key shape(N, key_len, heads, headdim)
        #query shape(N, query_len, heads, headdim)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-e20"))

        attention = torch.softmax(energy/(self.embedSize ** 0.5))
        out = torch.einsum("nhql, nlhd->nqhd",[attention, values]).reshape(
            N, que_len, self.headDim*self.heads
        )
        # attention shape :(N, heads, querLen, keyLen)
        #value shape:(N,value_len, heads, headidm)
        out = self.fc_out(out)
        return out

class transformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(transformerBlock, self).__init__()
        self.attention = selfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)#map back
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, val, key, que, mask):
        attention = self.attention(val,key,que,mask)
        x = self.dropout(self.norm1(attention+que))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward+x))

class encoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,#source vocabulary size
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length#用于位置embeding
    ):
        super(encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device#????
        self.word_embeding = nn.Embedding(src_vocab_size, embed_size)#numembeddings代表一共有多少个词, embedding_dim代表你想要为每个词创建一个多少维的向量来表示它，
        self.positionEmbeding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                transformerBlock(
                    embed_size,
                    heads,
                    dropout,
                    forward_expansion
                )
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0,seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.word_embeding(x) + self.positionEmbeding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

class decoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forwad_expansion, dropout, device):
        super(decoderBlock, self).__init__()
        self.attention = selfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = transformerBlock(
            embed_size,heads,dropout,forwad_expansion
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x,x,x,trg_mask)
        query = self.dropout(self.norm(attention+x))#decoder的query来自decoder的上一层
        out = self.transformer_block(value, key, query, src_mask)

        return out


class decoder(nn.Module):
    def __init__(self,
                 trg_vocab_size,
                 embed_size,
                 num_layer,
                 heads,
                 forward_expansion,
                 dropout,
                 device,
                 max_length,
    ):
        super(decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.psition_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                decoderBlock(embed_size, heads,forward_expansion,dropout, device)
                for _ in range(num_layer)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0,seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.psition_embedding(positions))
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)