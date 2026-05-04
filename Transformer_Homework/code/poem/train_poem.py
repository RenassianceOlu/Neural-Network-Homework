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

# ----------------- 1. 超参数设置 -----------------
BATCH_SIZE = 64
EPOCHS = 40
EMBEDDING_DIM = 512        # 词嵌入维度
NUM_HEADS = 8              # 多头注意力头数
NUM_LAYERS = 3             # Transformer 层数
HIDDEN_DIM = 512          # 前馈网络隐藏层维度
MAX_SEQ_LEN = 64           # 古诗最大长度
LEARNING_RATE = 0.001      # 学习率
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 特殊字符定义
START_TOKEN = '<START>'
END_TOKEN = '<END>'
PAD_TOKEN = '<PAD>'

# ----------------- 2. 数据处理与加载 -----------------
def load_and_filter_data(data_dir):
    """读取所有 JSON 并筛选出七言绝句"""
    poems = []
    file_pattern = os.path.join(data_dir, 'poet.song.*.json')
    files = glob.glob(file_pattern)
    
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                paragraphs = item.get('paragraphs', [])
                if len(paragraphs) == 2 and all(len(p) == 16 for p in paragraphs):
                    poem = "".join(paragraphs)
                    poems.append(poem)
    return poems

class PoetryDataset(Dataset):
    def __init__(self, poems):
        self.word2idx = {PAD_TOKEN: 0, START_TOKEN: 1, END_TOKEN: 2}
        for poem in poems:
            for word in poem:
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
                    
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
        self.data_sequences = []
        for poem in poems:
            seq = [self.word2idx[START_TOKEN]] + [self.word2idx[w] for w in poem] + [self.word2idx[END_TOKEN]]
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
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x

class PoetryTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_len):
        super(PoetryTransformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # norm_first=True (Pre-LN) 搭配初始化，极大加快收敛速度
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # 应用权重初始化，解决初始 Loss 太高的问题
        self._init_weights()

    def _init_weights(self):
        """Xavier 均匀分布初始化，让初始 Loss 稳定，并容易迅速收敛"""
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        seq_len = src.size(1)
        mask = self.generate_square_subsequent_mask(seq_len).to(src.device)
        
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        output = self.transformer(src, mask=mask) 
        output = self.fc_out(output)              
        return output

# ----------------- 4. 训练与图表绘制 -----------------
def train(dataset):
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = PoetryTransformer(dataset.vocab_size, EMBEDDING_DIM, NUM_HEADS, 
                              NUM_LAYERS, HIDDEN_DIM, MAX_SEQ_LEN).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    loss_history = []
    
    model.train()
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
            
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")
        
    torch.save(model.state_dict(), 'poetry_transformer.pth')

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), loss_history, marker='o', color='blue', label='Train Loss')
    
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(range(1, EPOCHS + 1))               
    plt.grid(True, linestyle='--', alpha=0.7)     
    plt.legend(loc='upper right')                  
    
    plt.savefig('loss_curve.png')
    plt.show()
    
    return model

# ----------------- 5. 使用 Top-K 与 Temperature 采样生成 -----------------
def generate_poetry(model, dataset, start_words="明月", temperature=0.8, top_k=5):
    """加入温度与 Top-k 采样，解决 Transformer 复读机问题"""
    model.eval()
    words = list(start_words)
    
    seq_indices = [dataset.word2idx[START_TOKEN]]
    for w in words:
        seq_indices.append(dataset.word2idx.get(w, dataset.word2idx[PAD_TOKEN]))
    
    punctuation_map = {7: '，', 15: '。', 23: '，', 31: '。'}
    
    with torch.no_grad():
        while len(words) < 32: 
            current_len = len(words)
            
            if current_len in punctuation_map:
                next_word = punctuation_map[current_len]
                top_index = dataset.word2idx.get(next_word, dataset.word2idx[PAD_TOKEN])
            else:
                x = torch.tensor([seq_indices], dtype=torch.long).to(DEVICE)
                output = model(x)
                logits = output[0, -1, :] 
                
                # 屏蔽模型提前预测出标点和结束符的可能
                logits[dataset.word2idx.get('，', 0)] = float('-inf')
                logits[dataset.word2idx.get('。', 0)] = float('-inf')
                logits[dataset.word2idx[END_TOKEN]] = float('-inf')
                
                # --- 重复字惩罚 ---
                if len(seq_indices) > 0:
                    last_idx = seq_indices[-1]
                    if logits[last_idx] > 0:
                        logits[last_idx] /= 1.5 
                    else:
                        logits[last_idx] *= 1.5
                
                # --- 温度缩放与 Top-K 采样 ---
                logits = logits / temperature
                top_logits, top_indices = torch.topk(logits, top_k)
                probabilities = F.softmax(top_logits, dim=-1)
                
                sampled_i = torch.multinomial(probabilities, 1).item()
                top_index = top_indices[sampled_i].item()
                
                next_word = dataset.idx2word.get(top_index, "")
                
            words.append(next_word)
            seq_indices.append(top_index)

    # 切分两行 (XX...XX。 XX...XX。)
    poem_str = "".join(words)
    formatted_poem = f"{poem_str[:16]}\n{poem_str[16:32]}"
    
    return formatted_poem

# ----------------- 主程序 -----------------
if __name__ == "__main__":
    current_dir = "."  
    print("正在加载和处理诗歌数据...")
    poems = load_and_filter_data(current_dir)
    print(f"成功加载七言绝句数量: {len(poems)}")
    
    dataset = PoetryDataset(poems)
    print(f"语料词汇表大小: {dataset.vocab_size}")
    
    print("开始训练 Transformer 模型...")
    model = train(dataset)
    
    print("\nTransformer 生成的古诗：")
    print("--------------------------------")
    generated_poem = generate_poetry(model, dataset, start_words="明月")
    print(generated_poem)
    print("--------------------------------")