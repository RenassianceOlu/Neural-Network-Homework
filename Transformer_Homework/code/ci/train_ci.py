import os
import glob
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ----------------- 1. 超参数设置  -----------------
BATCH_SIZE = 128       
EPOCHS = 100            
EMBEDDING_DIM = 256    
NUM_HEADS = 8          # 增加注意力头数，让模型捕捉词牌名和断句的细微规律
NUM_LAYERS = 4         
HIDDEN_DIM = 1024      
MAX_SEQ_LEN = 150      
LEARNING_RATE = 0.001  
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

START_TOKEN = '<START>'
END_TOKEN = '<END>'
PAD_TOKEN = '<PAD>'
SEP_TOKEN = '<SEP>'

# ----------------- 2. 数据处理与加载 -----------------
def load_ci_data(data_dir):
    """读取所有 宋词 JSON 文件"""
    cis = []
    file_pattern = os.path.join(data_dir, 'ci.song.*.json')
    files = glob.glob(file_pattern)
    
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                rhythmic = item.get('rhythmic', '')  
                paragraphs = item.get('paragraphs', []) 
                if rhythmic and paragraphs:
                    content = "".join(paragraphs)
                    if 10 <= len(content) <= 120:
                        cis.append((rhythmic, content))
    return cis

class CiDataset(Dataset):
    def __init__(self, cis):
        self.word2idx = {PAD_TOKEN: 0, START_TOKEN: 1, END_TOKEN: 2, SEP_TOKEN: 3}
        for rhythmic, content in cis:
            for word in rhythmic + content:
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
                    
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
        self.data_sequences = []
        for rhythmic, content in cis:
            seq = [self.word2idx[START_TOKEN]] + \
                  [self.word2idx[w] for w in rhythmic] + \
                  [self.word2idx[SEP_TOKEN]] + \
                  [self.word2idx[w] for w in content] + \
                  [self.word2idx[END_TOKEN]]
            
            if len(seq) > MAX_SEQ_LEN:
                seq = seq[:MAX_SEQ_LEN]
            else:
                seq = seq + [self.word2idx[PAD_TOKEN]] * (MAX_SEQ_LEN - len(seq))
            self.data_sequences.append(seq)
            
    def __len__(self):
        return len(self.data_sequences)
        
    def __getitem__(self, idx):
        seq = self.data_sequences[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y

# ----------------- 3. 构建 Transformer 模型 -----------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0) 

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class CiTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_len):
        super(CiTransformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, 5000)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            batch_first=True,
            norm_first=True,
            dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    def forward(self, src):
        mask = self.generate_square_subsequent_mask(src.size(1)).to(src.device)
        src = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        output = self.transformer(src, mask=mask) 
        return self.fc_out(output)

# ----------------- 4. 训练与收敛追踪 -----------------
def train(dataset):
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = CiTransformer(dataset.vocab_size, EMBEDDING_DIM, NUM_HEADS, 
                          NUM_LAYERS, HIDDEN_DIM, MAX_SEQ_LEN).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.word2idx[PAD_TOKEN])
    loss_history = []
    
    model.train()
    print("====== 开启宋词生成模型训练 ======")
    for epoch in range(EPOCHS):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.reshape(-1, dataset.vocab_size), y.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()  # 步进学习率衰减
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
    torch.save(model.state_dict(), 'songci_transformer.pth')

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), loss_history, marker='o', color='blue', label='Train Loss')
    plt.title("Enhanced Ci-Transformer Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(range(1, EPOCHS + 1, max(1, EPOCHS//10)))
    plt.grid(True, linestyle='--', alpha=0.7)     
    plt.legend(loc='upper right')                  
    plt.savefig('ci_loss_curve.png')
    plt.show()
    return model

# ----------------- 5. 指定首字和词牌名生成 -----------------
def generate_ci_with_start(model, dataset, ci_pai_ming="水调歌头", start_words="明月", max_len=120, temperature=0.8, top_k=5):

    model.eval()
    
    # prompt ：<START> 词牌名 <SEP> 明月
    prompt_words = [START_TOKEN] + list(ci_pai_ming) + [SEP_TOKEN] + list(start_words)
    seq_indices = [dataset.word2idx.get(w, dataset.word2idx[PAD_TOKEN]) for w in prompt_words]
    
    generated_words = []
    
    with torch.no_grad():
        for _ in range(max_len): 
            x = torch.tensor([seq_indices[-MAX_SEQ_LEN:]] if len(seq_indices) > MAX_SEQ_LEN else [seq_indices], dtype=torch.long).to(DEVICE)
            output = model(x)
            logits = output[0, -1, :] 
            
            logits[dataset.word2idx[PAD_TOKEN]] = float('-inf')
            logits[dataset.word2idx[START_TOKEN]] = float('-inf')
            logits[dataset.word2idx[SEP_TOKEN]] = float('-inf')
            
            if len(seq_indices) > 0:
                last_idx = seq_indices[-1]
                if logits[last_idx] > 0:
                    logits[last_idx] /= 1.5 
                else:
                    logits[last_idx] *= 1.5
            
            logits = logits / temperature
            top_logits, top_indices = torch.topk(logits, top_k)
            probabilities = F.softmax(top_logits, dim=-1)
            
            sampled_i = torch.multinomial(probabilities, 1).item()
            top_index = top_indices[sampled_i].item()
            
            if top_index == dataset.word2idx[END_TOKEN]:
                break
                
            next_word = dataset.idx2word.get(top_index, "")
            generated_words.append(next_word)
            seq_indices.append(top_index)

    return f"【{ci_pai_ming}】\n{start_words}" + "".join(generated_words)

if __name__ == "__main__":
    data_directory = "."  
    cis_data = load_ci_data(data_directory)
    #cis_data = load_ci_data(data_directory)[:3000]  
    if len(cis_data) == 0:
        print("【错误】请确保你在 `ci` 目录下运行，能找到 ci.song.*.json")
    else:
        print(f"成功加载宋词: {len(cis_data)} 首")
        dataset = CiDataset(cis_data)
        
        model = train(dataset)
        
        print("\n============ 成果展示 ============\n")
        print(generate_ci_with_start(model, dataset, ci_pai_ming="浣溪沙", start_words="明月"))
        print("\n")
        print(generate_ci_with_start(model, dataset, ci_pai_ming="如梦令", start_words="明月"))
        print("\n==================================")