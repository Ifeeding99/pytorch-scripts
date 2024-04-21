import torch
import torch.nn as nn


class MHA(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        assert(embed_dim % n_heads == 0), "embedding dimension should be a multiple of number of heads"


        self.queries_matrix = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.keys_matrix = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.values_matrix = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.fc = nn.Linear(self.n_heads * self.head_dim, embed_dim)

    def forward(self, queries, keys, values, mask = None): #shape: batch,length, embedding_dim
        q = queries.reshape(queries.shape[0],queries.shape[1], self.n_heads, self.head_dim)
        k = keys.reshape(keys.shape[0], keys.shape[1], self.n_heads, self.head_dim)
        v = values.reshape(values.shape[0], values.shape[1], self.n_heads, self.head_dim)

        q = self.queries_matrix(q)
        k = self.keys_matrix(k)
        v = self.values_matrix(v)

        q_v_matrix = torch.einsum("bqhd,bkhd -> bhqk", q,k) # b=batch, q = queries length, h = number of heads, d = head's dimension
        if mask is not None:
            q_v_matrix.masked_fill(mask == 0, 1e-20) # where mask == 0 put negative infinity (used for decoder)

        attention = nn.functional.softmax(q_v_matrix/(k.shape[1])**(-1/2), dim = -1)
        out = torch.einsum("bhqk,bkhd -> bqhd", attention, v) # b=n batch, h=number of heads, q=queries length, k=key length, v= values length, d= head's dimension
        # note that for the values I have written "bkhd" beacause the lengths of the values, the queries and the keys are all the same
        # and I want to multiply along that dimension
        out = out.reshape(out.shape[0], out.shape[1], self.n_heads*self.head_dim) # concatenating from all heads, n_heads*head_dim = embedding size
        out = self.fc(out)
        return out

class Encoder_Block(nn.Module):
    def __init__(self, embed_size, n_heads, forward_expansion, dropout):
        super().__init__()
        self.embed_size = embed_size
        self.n_heads = n_heads
        self.forward_expansion = forward_expansion
        self.attention = MHA(self.embed_size, n_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_size, self.forward_expansion*self.embed_size),
            nn.ReLU(),
            nn.Linear(self.forward_expansion*self.embed_size, embed_size)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, queries, keys, values, mask):
        att = self.attention(queries,keys, values, mask)
        add_and_norm = self.norm1(att + queries)
        out = self.drop(add_and_norm)
        x = self.ffn(out)
        add_and_norm2 = self.norm2(x+out)
        out = self.drop(add_and_norm2)
        return out

class Encoder(nn.Module):
    def __init__(self, vocab_size, n_heads, embedding_dim, forward_exp, n_layers, dropout, max_length):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.forward_expansion = forward_exp
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.max_length = max_length
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.position_embedding = nn.Embedding(self.max_length, self.embedding_dim)

        self.layers = nn.ModuleList([
            Encoder_Block(self.embedding_dim, self.n_heads, self.forward_expansion,dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, mask):
        n,l = x.shape[0],x.shape[1]
        positions = torch.arange(0,l).expand(n,l)
        emb = self.word_embedding(x) + self.position_embedding(positions)
        x = self.dropout(emb)
        for layer in self.layers:
            x = layer(x,x,x, mask)
        return x


class Decoder_Block(nn.Module):
    def __init__(self, embed_size, forward_expansion, n_heads, dropout):
        super().__init__()
        self.embed_size = embed_size
        self.forward_expansion = forward_expansion
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.attention = MHA(self.embed_size, self.n_heads)
        self.norm = nn.LayerNorm(self.embed_size)
        self.block = Encoder_Block(self.embed_size, self.n_heads, self.forward_expansion, dropout)


    def forward(self, x,v,k, src_mask, trg_mask):
        attention = self.attention(x,x,x, mask=trg_mask)
        q = self.norm(attention+x)
        q = self.dropout(q)
        out = self.block(q,k,v,src_mask)
        return out

class Decoder(nn.Module):
    def __init__(self, embed_size, num_layers, n_heads, trg_vocab_size, forward_expansion, dropout, max_length):
        super().__init__()
        self.word_embeddings = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(max_length, embed_size)
        self.blocks = nn.ModuleList([
            Decoder_Block(embed_size, forward_expansion,n_heads, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_size, trg_vocab_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        n,l = x.shape[0], x.shape[1]
        pos = torch.arange(0,l).expand(n,l)
        emb = self.word_embeddings(x)+self.position_embeddings(pos)
        x = self.drop(emb)
        for layer in self.blocks:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        x = self.fc(x)
        return x



class Transformer(nn.Module):
    def __init__(self, src_vocab_size,
                 trg_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 emb_dim = 256,
                 num_layers=6,
                 forward_expansion = 4,
                 heads = 8,
                 dropout = 0.5,
                 max_length = 100):
        super().__init__()
        self.encoder = Encoder(src_vocab_size,heads,emb_dim,forward_expansion,num_layers,dropout,max_length)
        self.decoder = Decoder(emb_dim, num_layers, heads, trg_vocab_size, forward_expansion,dropout, max_length)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx


    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        n, trg_len = trg.shape[0], trg.shape[1]
        trg_mask = torch.tril(torch.ones((trg_len,trg_len))).expand(n,1,trg_len,trg_len)
        return trg_mask

    def forward(self, src, target):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(target)
        enc = self.encoder(src, src_mask)
        dec = self.decoder(target, enc, src_mask, trg_mask)
        return dec

if __name__ == "__main__":
    x = torch.Tensor([[1,5,6,4,3,9,5,2,0],[1,8,7,3,4,5,6,7,2]]).long()
    trg = torch.Tensor([[1,7,4,3,5,9,2,0],[1,5,6,2,4,7,6,2]]).long()
    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx)
    out = model(x, trg[:,:-1])
    print(out.shape)



