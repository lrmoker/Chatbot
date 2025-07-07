# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import logging
import os
import json
from model.pre_process import Corpus
from model.model import Seq2Seq
import sys
import time
import argparse
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict

# 确保必要的目录存在
os.makedirs('data', exist_ok=True)
os.makedirs('model', exist_ok=True)

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/chatbot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def check_data_format(data):
    """检查数据格式是否正确"""
    if not isinstance(data, list):
        logging.error("数据格式错误：应为列表格式")
        return False
    
    if len(data) == 0:
        logging.error("数据为空")
        return False
    
    for item in data[:5]:  # 检查前5条数据
        if not isinstance(item, dict) or 'question' not in item or 'answer' not in item:
            logging.error("数据格式错误：每条数据应包含 'question' 和 'answer' 字段")
            return False
    
    return True

class ChatDataset(Dataset):
    def __init__(self, corpus, max_len=50):
        self.corpus = corpus
        self.max_len = max_len
        
    def __len__(self):
        return len(self.corpus.trainingSamples)
        
    def __getitem__(self, idx):
        sample = self.corpus.trainingSamples[idx]
        input_ids = sample[0][:self.max_len]
        target_ids = sample[1][:self.max_len]
        
        # 确保序列长度一致
        if len(input_ids) < self.max_len:
            input_ids = input_ids + [0] * (self.max_len - len(input_ids))
        if len(target_ids) < self.max_len:
            target_ids = target_ids + [0] * (self.max_len - len(target_ids))
            
        return {
            'input_ids': torch.tensor(input_ids),
            'target_ids': torch.tensor(target_ids)
        }

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 数据预处理类
class Preprocessor:
    def __init__(self, min_freq=2, max_vocab_size=308):
        self.word2id = {'<PAD>': 0, '< SOS >': 1, '<EOS>': 2, '<UNK>': 3}
        self.id2word = {0: '<PAD>', 1: '< SOS >', 2: '<EOS>', 3: '<UNK>'}
        self.word_freq = {}
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        
    def build_vocab(self, conversations):
        # 统计字符频率
        for conv in conversations:
            for text in [conv['question'], conv['answer']]:
                for char in text:
                    if char not in self.word_freq:
                        self.word_freq[char] = 0
                    self.word_freq[char] += 1
        
        # 按频率排序并限制词表大小
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)
        idx = len(self.word2id)
        for char, freq in sorted_words:
            if freq >= self.min_freq and char not in self.word2id:
                if idx >= self.max_vocab_size:
                    break
                self.word2id[char] = idx
                self.id2word[idx] = char
                idx += 1
    
    def text_to_ids(self, text):
        return [self.word2id.get(char, self.word2id['<UNK>']) for char in text]
    
    def save_vocab(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.word2id, f, ensure_ascii=False, indent=2)

# 数据加载器
class DataLoader:
    def __init__(self, conversations, preprocessor, batch_size=32, max_length=50, device='cpu'):
        self.conversations = conversations
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.max_length = max_length
        self.n_conversations = len(conversations)
        self.idx = 0
        self.device = device
    
    def __len__(self):
        """返回批次数"""
        return (self.n_conversations + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        self.idx = 0
        return self
    
    def __next__(self):
        if self.idx >= self.n_conversations:
            raise StopIteration
        
        batch_input = []
        batch_target = []
        
        for _ in range(self.batch_size):
            if self.idx >= self.n_conversations:
                break
                
            conv = self.conversations[self.idx]
            input_text = conv['question']
            target_text = conv['answer']
            
            # 转换为ID序列
            input_ids = [self.preprocessor.word2id['< SOS >']]
            input_ids.extend(self.preprocessor.text_to_ids(input_text))
            input_ids.append(self.preprocessor.word2id['<EOS>'])
            
            target_ids = [self.preprocessor.word2id['< SOS >']]
            target_ids.extend(self.preprocessor.text_to_ids(target_text))
            target_ids.append(self.preprocessor.word2id['<EOS>'])
            
            # 截断或填充
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length-1] + [self.preprocessor.word2id['<EOS>']]
            else:
                input_ids.extend([self.preprocessor.word2id['<PAD>']] * (self.max_length - len(input_ids)))
                
            if len(target_ids) > self.max_length:
                target_ids = target_ids[:self.max_length-1] + [self.preprocessor.word2id['<EOS>']]
            else:
                target_ids.extend([self.preprocessor.word2id['<PAD>']] * (self.max_length - len(target_ids)))
            
            batch_input.append(input_ids)
            batch_target.append(target_ids)
            
            self.idx += 1
            
        if not batch_input:
            raise StopIteration
            
        # 将数据移动到指定设备
        return (torch.LongTensor(batch_input).to(self.device), 
                torch.LongTensor(batch_target).to(self.device))

def load_json_conversations(file_path):
    """加载JSON格式的对话数据"""
    try:
        logging.info(f"正在加载文件: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # 记录数据格式信息
            logging.info(f"数据格式: {type(data)}")
            if isinstance(data, dict):
                logging.info(f"字典键: {list(data.keys())}")
            elif isinstance(data, list):
                logging.info(f"列表长度: {len(data)}")
                if len(data) > 0:
                    logging.info(f"第一个元素类型: {type(data[0])}")
                    if isinstance(data[0], dict):
                        logging.info(f"第一个元素的键: {list(data[0].keys())}")
            
            # 处理列表中的列表格式（每个内部列表包含对话）
            if isinstance(data, list):
                conversations = []
                for dialog in data:
                    if isinstance(dialog, list) and len(dialog) >= 2:
                        # 将对话中的每对相邻语句作为问答对
                        for i in range(len(dialog) - 1):
                            conversations.append({
                                'question': dialog[i],
                                'answer': dialog[i + 1]
                            })
                logging.info(f"从对话列表格式加载了 {len(conversations)} 条对话")
                return conversations
            
            # 处理字典格式（包含conversations字段）
            elif isinstance(data, dict) and 'conversations' in data:
                conversations = []
                for conv in data['conversations']:
                    if 'input' in conv and 'output' in conv:
                        conversations.append({
                            'question': conv['input'],
                            'answer': conv['output']
                        })
                logging.info(f"从字典格式加载了 {len(conversations)} 条对话")
                return conversations
            
            logging.error(f"数据格式错误：{file_path} 中的数据格式不支持")
            return []
            
    except json.JSONDecodeError as e:
        logging.error(f"JSON解析错误：{str(e)}")
        return []
    except FileNotFoundError:
        logging.error(f"文件不存在：{file_path}")
        return []
    except Exception as e:
        logging.error(f"加载文件 {file_path} 时出错: {str(e)}")
        return []

def calculate_metrics(predictions, targets, ignore_index=0):
    """计算评估指标"""
    # 过滤掉padding token
    mask = targets != ignore_index
    predictions = predictions[mask]
    targets = targets[mask]
    
    if len(targets) == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
    
    # 计算准确率
    accuracy = (predictions == targets).float().mean().item()
    
    # 计算精确率、召回率和F1分数
    true_positives = ((predictions == targets) & (predictions != ignore_index)).sum().item()
    predicted_positives = (predictions != ignore_index).sum().item()
    actual_positives = (targets != ignore_index).sum().item()
    
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
    recall = true_positives / actual_positives if actual_positives > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def save_epoch_metrics(metrics, epoch, total_epochs, phase, device):
    """保存每个epoch结束时的指标到日志文件"""
    log_file = "data/epoch_metrics.log"
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"Epoch时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{phase} Epoch: {epoch + 1}/{total_epochs}\n")
        f.write("-"*80 + "\n")
        f.write(f"{'指标':<10} {'值':<15}\n")
        f.write("-"*80 + "\n")
        f.write(f"{'损失':<10} {metrics['loss']:<15.4f}\n")
        f.write(f"{'准确率':<10} {metrics['accuracy']:<15.4f}\n")
        f.write(f"{'精确率':<10} {metrics['precision']:<15.4f}\n")
        f.write(f"{'召回率':<10} {metrics['recall']:<15.4f}\n")
        f.write(f"{'F1分数':<10} {metrics['f1']:<15.4f}\n")
        if device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
            memory_cached = torch.cuda.memory_reserved(device) / 1024**3
            f.write("-"*80 + "\n")
            f.write(f"GPU内存使用: {memory_allocated:.2f}GB / {memory_cached:.2f}GB\n")
        f.write("="*80 + "\n\n")

def train(model, train_loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0
    metrics = defaultdict(float)
    num_batches = len(train_loader)
    
    for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Use mixed precision training if scaler is provided
        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(input_seq, target_seq)
                loss = criterion(output.view(-1, output.size(-1)), target_seq.view(-1))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(input_seq, target_seq)
            loss = criterion(output.view(-1, output.size(-1)), target_seq.view(-1))
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate metrics
        with torch.no_grad():
            predictions = output.argmax(dim=-1)
            batch_metrics = calculate_metrics(predictions, target_seq)
            for metric, value in batch_metrics.items():
                metrics[metric] += value
        
        # Log progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            logging.info(f'Train Batch [{batch_idx+1}/{num_batches}] Loss: {avg_loss:.4f}')
            
            # Log GPU memory if using CUDA
            if device == 'cuda':
                allocated = torch.cuda.memory_allocated() / 1024**2
                cached = torch.cuda.memory_reserved() / 1024**2
                logging.info(f'GPU Memory: {allocated:.0f}MB allocated, {cached:.0f}MB cached')
    
    # Calculate average metrics
    avg_loss = total_loss / num_batches
    avg_metrics = {metric: value / num_batches for metric, value in metrics.items()}
    
    return avg_loss, avg_metrics

def validate(model, val_loader, criterion, device, scaler=None):
    model.eval()
    total_loss = 0
    metrics = defaultdict(float)
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch_idx, (input_seq, target_seq) in enumerate(val_loader):
            # Use mixed precision if scaler is provided
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    output = model(input_seq, target_seq)
                    loss = criterion(output.view(-1, output.size(-1)), target_seq.view(-1))
            else:
                output = model(input_seq, target_seq)
                loss = criterion(output.view(-1, output.size(-1)), target_seq.view(-1))
            
            total_loss += loss.item()
            
            # Calculate metrics
            predictions = output.argmax(dim=-1)
            batch_metrics = calculate_metrics(predictions, target_seq)
            for metric, value in batch_metrics.items():
                metrics[metric] += value
            
            # Log progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                logging.info(f'Val Batch [{batch_idx+1}/{num_batches}] Loss: {avg_loss:.4f}')
                
                # Log GPU memory if using CUDA
                if device == 'cuda':
                    allocated = torch.cuda.memory_allocated() / 1024**2
                    cached = torch.cuda.memory_reserved() / 1024**2
                    logging.info(f'GPU Memory: {allocated:.0f}MB allocated, {cached:.0f}MB cached')
    
    # Calculate average metrics
    avg_loss = total_loss / num_batches
    avg_metrics = {metric: value / num_batches for metric, value in metrics.items()}
    
    return avg_loss, avg_metrics

def plot_training_metrics(metrics_history, save_path):
    """绘制训练指标图表"""
    plt.figure(figsize=(15, 10))
    
    # 绘制损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(metrics_history['train_loss'], label='Training Loss')
    plt.plot(metrics_history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(metrics_history['train_accuracy'], label='Training Accuracy')
    plt.plot(metrics_history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 绘制F1分数曲线
    plt.subplot(2, 2, 3)
    plt.plot(metrics_history['train_f1'], label='Training F1')
    plt.plot(metrics_history['val_f1'], label='Validation F1')
    plt.title('F1 Score Curves')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    # 绘制精确率和召回率曲线
    plt.subplot(2, 2, 4)
    plt.plot(metrics_history['train_precision'], label='Training Precision')
    plt.plot(metrics_history['train_recall'], label='Training Recall')
    plt.plot(metrics_history['val_precision'], label='Validation Precision')
    plt.plot(metrics_history['val_recall'], label='Validation Recall')
    plt.title('Precision and Recall Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_model(train_data, val_data, model_path, vocab_path, device='cpu', 
             batch_size=32, epochs=10, learning_rate=0.001, scaler=None):
    """
    Train the model using the provided data
    """
    # Set up GPU if available
    if device == 'cuda' and torch.cuda.is_available():
        # Enable cuDNN auto-tuner
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # Configure memory fraction to avoid OOM
        torch.cuda.set_per_process_memory_fraction(0.8)
        
        # Log GPU info
        gpu_name = torch.cuda.get_device_name()
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
        logging.info(f'Using GPU: {gpu_name} with {total_memory:.0f}MB total memory')
    else:
        device = 'cpu'
        logging.info('Using CPU for training')

    # 创建预处理器
    preprocessor = Preprocessor()
    preprocessor.build_vocab(train_data)
    preprocessor.save_vocab(vocab_path)
    
    # 创建数据加载器
    train_loader = DataLoader(
        conversations=train_data,
        preprocessor=preprocessor,
        batch_size=batch_size,
        max_length=50,
        device=device
    )
    
    val_loader = DataLoader(
        conversations=val_data,
        preprocessor=preprocessor,
        batch_size=batch_size,
        max_length=50,
        device=device
    )
    
    vocab_size = len(preprocessor.word2id)
    logging.info(f'Vocabulary size: {vocab_size}')

    # Initialize model, loss function and optimizer
    model = Seq2Seq(
        dataClass=preprocessor,
        featureSize=256,
        hiddenSize=256,
        numLayers=3,
        dropout=0.1,
        modelType='lstm'
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    history = defaultdict(list)
    
    try:
        for epoch in range(epochs):
            logging.info(f'\nEpoch {epoch+1}/{epochs}')
            
            # Training phase
            train_loss, train_metrics = train(
                model, train_loader, optimizer, criterion, device, scaler
            )
            history['train_loss'].append(train_loss)
            for metric, value in train_metrics.items():
                history[f'train_{metric}'].append(value)
            
            # Validation phase
            val_loss, val_metrics = validate(
                model, val_loader, criterion, device, scaler
            )
            history['val_loss'].append(val_loss)
            for metric, value in val_metrics.items():
                history[f'val_{metric}'].append(value)
            
            # Log epoch metrics
            logging.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            for metric in train_metrics:
                logging.info(f'{metric}: Train={train_metrics[metric]:.4f}, Val={val_metrics[metric]:.4f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'epoch': epoch
                }, model_path)
                logging.info(f'Model saved to {model_path}')
            
            # Clear GPU cache if using CUDA
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    except KeyboardInterrupt:
        logging.info('Training interrupted by user')
    
    # Plot and save training metrics
    plot_training_metrics(history, os.path.join(os.path.dirname(model_path), 'training_metrics.png'))
    
    return model, history

def main():
    # 设置参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        # 显示GPU信息
        logging.info(f"GPU设备: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU内存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        # 设置GPU内存分配策略
        torch.cuda.set_per_process_memory_fraction(0.8)  # 使用80%的GPU内存
        torch.backends.cudnn.benchmark = True  # 启用cuDNN自动调优
        torch.backends.cudnn.enabled = True    # 启用cuDNN
        # 设置默认设备
        torch.cuda.set_device(0)
        # 启用自动混合精度训练
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    batch_size = 64
    epochs = 30
    learning_rate = 0.001
    feature_size = 256
    hidden_size = 256
    num_layers = 3
    dropout = 0.1
    max_length = 50
    
    # 记录训练参数
    logging.info("训练参数配置:")
    logging.info(f"批大小: {batch_size}")
    logging.info(f"训练轮数: {epochs}")
    logging.info(f"学习率: {learning_rate}")
    logging.info(f"特征维度: {feature_size}")
    logging.info(f"隐藏层维度: {hidden_size}")
    logging.info(f"层数: {num_layers}")
    logging.info(f"Dropout率: {dropout}")
    logging.info(f"最大序列长度: {max_length}")
    
    # 设置随机种子
    set_seed(42)
    logging.info("随机种子已设置为42")
    
    # 加载数据
    logging.info("开始加载训练数据...")
    train_data = load_json_conversations('data/LCCC-base_train.json')
    if not train_data:
        logging.error("无法加载训练数据，程序退出")
        return
    logging.info(f"成功加载训练数据，共 {len(train_data)} 条对话")
    
    logging.info("开始加载验证数据...")
    val_data = load_json_conversations('data/LCCC-base_test.json')
    if not val_data:
        logging.error("无法加载验证数据，程序退出")
        return
    logging.info(f"成功加载验证数据，共 {len(val_data)} 条对话")
    
    # 训练模型
    logging.info("开始训练模型...")
    train_model(
        train_data=train_data,
        val_data=val_data,
        model_path='checkpoints/best_model.pth',
        vocab_path='checkpoints/vocab.json',
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        scaler=scaler
    )
    logging.info("模型训练完成！")

if __name__ == "__main__":
    main()

print(torch.cuda.get_device_name(0))  # 如果成功，会显示GPU型号 