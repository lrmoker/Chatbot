# 导入库
import math
import json
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from matplotlib import pyplot as plt
from matplotlib import ticker
from nltk.translate.bleu_score import sentence_bleu
import time
import random
import os
import jieba
import logging
import numpy as np
import pandas as pd
import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from itertools import chain
import re
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

jieba.setLogLevel(logging.INFO)

# 定义开始符和结束符
sosToken = 1
eosToken = 0

# CNN编码器


class CNNEncoder(nn.Module):
    def __init__(self, inputSize, featureSize, hiddenSize, numLayers=3, dropout=0.1):
        super(CNNEncoder, self).__init__()
        self.embedding = nn.Embedding(inputSize, featureSize, padding_idx=0)

        # CNN层
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(featureSize if i == 0 else hiddenSize, hiddenSize,
                          kernel_size=3, padding=1),
                nn.BatchNorm1d(hiddenSize),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for i in range(numLayers)
        ])

        # 位置编码
        self.position_embedding = nn.Parameter(
            torch.randn(1, 512, featureSize))

        # 保存参数
        self.featureSize = featureSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers

    def forward(self, input_seq, lengths=None, hidden=None):
        # 输入编码: [batch, seq_len] -> [batch, seq_len, feature_size]
        embedded = self.embedding(input_seq)

        # 添加位置编码
        seq_len = embedded.size(1)
        embedded = embedded + self.position_embedding[:, :seq_len, :]

        # CNN处理: [batch, feature_size, seq_len]
        x = embedded.transpose(1, 2)

        # 通过多层CNN
        for conv in self.convs:
            x = conv(x)

        # 转回 [batch, seq_len, hidden_size]
        output = x.transpose(1, 2)

        return output, output[:, -1, :].unsqueeze(0)

# CNN解码器


class CNNDecoderRNN(nn.Module):
    def __init__(self, outputSize, featureSize, hiddenSize, numLayers=3, dropout=0.1):
        super(CNNDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(outputSize, featureSize, padding_idx=0)

        # CNN层
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(featureSize if i == 0 else hiddenSize, hiddenSize,
                          kernel_size=3, padding=1),
                nn.BatchNorm1d(hiddenSize),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for i in range(numLayers)
        ])

        # 注意力层
        self.attention = nn.Linear(hiddenSize * 2, hiddenSize)

        # 输出层
        self.out = nn.Linear(hiddenSize, outputSize)

        # 位置编码
        self.position_embedding = nn.Parameter(
            torch.randn(1, 512, featureSize))

    def forward(self, input_step, hidden, encoder_outputs):
        batch_size = input_step.size(0)

        # 输入编码
        embedded = self.embedding(input_step)  # [batch, 1, feature_size]

        # 添加位置编码
        embedded = embedded + self.position_embedding[:, :embedded.size(1), :]

        # CNN处理
        x = embedded.transpose(1, 2)  # [batch, feature_size, 1]

        for conv in self.convs:
            x = conv(x)

        # 转回 [batch, 1, hidden_size]
        decoder_output = x.transpose(1, 2)

        # 计算注意力
        attn_weights = torch.bmm(
            decoder_output, encoder_outputs.transpose(1, 2))
        attn_weights = F.softmax(attn_weights, dim=2)

        # 应用注意力
        # [batch, 1, hidden_size]
        context = torch.bmm(attn_weights, encoder_outputs)

        # 合并上下文向量和解码器输出
        # [batch, 1, hidden_size*2]
        output = torch.cat((decoder_output, context), dim=2)
        output = self.attention(output)  # [batch, 1, hidden_size]
        output = torch.tanh(output)

        # 生成最终输出
        output = self.out(output)  # [batch, 1, vocab_size]
        output = F.log_softmax(output, dim=2)

        return output, decoder_output, attn_weights

# 定义核心类seq2seq


class Seq2Seq(nn.Module):
    def __init__(self, dataClass, featureSize=256, hiddenSize=256, numLayers=2,
                 dropout=0.1, modelType='cnn'):
        super(Seq2Seq, self).__init__()

        self.vocab_size = len(dataClass.word2id)
        self.modelType = modelType.lower()

        # 添加CUDA可用性检查
        self.cuda_available = torch.cuda.is_available()

        # 添加特殊token
        self.sosToken = dataClass.word2id.get('< SOS >', 1)
        self.eosToken = dataClass.word2id.get('<EOS>', 2)
        self.padToken = dataClass.word2id.get('<PAD>', 0)
        self.unkToken = dataClass.word2id.get('<UNK>', 3)
        self.word2id = dataClass.word2id

        # 创建共享的嵌入层
        self.embedding = nn.Embedding(
            self.vocab_size, featureSize, padding_idx=0)

        # 创建编码器和解码器
        self.encoder = CNNEncoder(
            inputSize=self.vocab_size,
            featureSize=featureSize,
            hiddenSize=hiddenSize,
            numLayers=numLayers,
            dropout=dropout
        )

        self.decoder = CNNDecoderRNN(
            outputSize=self.vocab_size,
            featureSize=featureSize,
            hiddenSize=hiddenSize,
            numLayers=numLayers,
            dropout=dropout
        )

    def forward(self, input, target, teacher_forcing_ratio=0.5):
        batch_size = input.size(0)
        max_len = target.size(1)
        vocab_size = self.vocab_size

        # Initialize outputs tensor
        outputs = torch.zeros(batch_size, max_len, vocab_size).to(input.device)

        # Get encoder outputs
        encoder_outputs, encoder_hidden = self.encoder(input)

        # First input to the decoder is the SOS token
        decoder_input = target[:, 0].unsqueeze(1)  # Shape: [batch_size, 1]
        decoder_hidden = encoder_hidden

        # Teacher forcing: Feed the target as the next input
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            for t in range(1, max_len):
                decoder_output, decoder_hidden, _ = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                outputs[:, t] = decoder_output.squeeze(1)
                decoder_input = target[:, t].unsqueeze(
                    1)  # Next input is current target
        else:
            for t in range(1, max_len):
                decoder_output, decoder_hidden, _ = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                outputs[:, t] = decoder_output.squeeze(1)
                # Next input is decoder's own output
                _, topi = decoder_output.squeeze(1).topk(1)
                decoder_input = topi.detach()  # Detach from history as input

        return outputs

    def predictByBeamSearch(self, input_seq, input_length, maxAnswerLength=20, beamWidth=3):
        """使用集束搜索生成回复"""
        with torch.no_grad():
            # 编码器前向传播
            encoderOutputs, encoderHidden = self.encoder(input_seq)

            # 准备解码器输入
            batchSize = input_seq.size(0)
            decoderInput = torch.LongTensor(
                [[self.sosToken]] * batchSize).to(input_seq.device)
            decoderHidden = encoderHidden

            # 初始化beam搜索
            beams = [([], 0.0)]  # (序列, 分数)

            # beam搜索主循环
            for _ in range(maxAnswerLength):
                candidates = []

                # 扩展每个当前序列
                for sequence, score in beams:
                    if sequence and sequence[-1] == self.eosToken:
                        candidates.append((sequence, score))
                        continue

                    # 准备当前输入
                    if not sequence:
                        currentInput = decoderInput
                    else:
                        currentInput = torch.LongTensor(
                            [[sequence[-1]]]).to(input_seq.device)

                    # 解码器前向传播
                    decoderOutput, decoderHidden, _ = self.decoder(
                        currentInput, decoderHidden, encoderOutputs)

                    # 获取概率分布
                    logits = decoderOutput[0, 0]
                    probs = F.softmax(logits, dim=0)

                    # 过滤掉特殊token和重复字符
                    if len(sequence) >= 2:
                        # 过滤掉特殊token
                        probs[self.sosToken] = 0
                        probs[self.padToken] = 0
                        probs[self.unkToken] = 0

                        # 防止重复字符
                        if sequence[-1] == sequence[-2]:
                            probs[sequence[-1]] = 0

                    # 获取top k个候选
                    topProbs, topTokens = probs.topk(
                        min(beamWidth, (probs > 0).sum().item()))

                    for prob, token in zip(topProbs, topTokens):
                        if prob.item() == 0:
                            continue

                        newSequence = sequence + [token.item()]
                        newScore = score + math.log(prob.item())

                        # 如果生成了EOS，增加额外奖励
                        if token.item() == self.eosToken:
                            newScore += 2.0

                        # 添加长度惩罚
                        lengthPenalty = (
                            (5.0 + len(newSequence)) ** 0.9) / (5.0 ** 0.9)
                        normalizedScore = newScore / lengthPenalty

                        candidates.append((newSequence, normalizedScore))

                # 如果没有候选，提前结束
                if not candidates:
                    break

                # 按分数排序并选择top k
                candidates.sort(key=lambda x: x[1], reverse=True)
                beams = candidates[:beamWidth]

                # 如果最佳候选以EOS结束，提前结束
                if beams[0][0][-1] == self.eosToken:
                    break

            # 选择最佳序列
            bestSequence = beams[0][0]

            # 如果序列没有以EOS结束，添加EOS
            if bestSequence[-1] != self.eosToken:
                bestSequence.append(self.eosToken)

            return bestSequence


class ChatBot:
    def __init__(self, model_path, vocab_path, device='cpu',
                 temperature=0.8, top_k=5, min_length=5, max_length=50,
                 ngram_size=3):
        # 加载词汇表
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.word2id = json.load(f)
            # 只保留前308个词
            self.word2id = dict(list(self.word2id.items())[:308])
            self.id2word = {int(v): k for k, v in self.word2id.items()}

        # 创建模型
        self.model = Seq2Seq(
            dataClass=type('DataClass', (), {
                'wordNum': len(self.word2id),
                'word2id': self.word2id,
                'id2word': self.id2word
            })(),
            featureSize=256,
            hiddenSize=256,  # 保持原有大小
            numLayers=3,     # 保持原有层数
            dropout=0.1,     # 保持原有dropout
            modelType='cnn'
        )

        # 加载模型参数
        logging.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 兼容旧格式
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.model.to(device)
        logging.info("Model loaded successfully")

        # 设置参数
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        self.min_length = min_length
        self.max_length = max_length
        self.ngram_size = ngram_size

        # 特殊token
        self.sos_token = self.word2id['< SOS >']
        self.eos_token = self.word2id['<EOS>']
        self.pad_token = self.word2id['<PAD>']
        self.unk_token = self.word2id['<UNK>']

        # 添加默认回复
        self.default_responses = [
            "抱歉，我现在无法回答这个问题。",
            "这个问题有点复杂，让我想想怎么回答。",
            "我需要更多信息来回答这个问题。",
            "这个问题超出了我的能力范围。",
            "让我换个方式回答这个问题。"
        ]

        logging.info(
            f"ChatBot initialized with temperature={temperature}, top_k={top_k}")

    def check_ngram_repeat(self, recent_chars, token, n):
        """检查n-gram重复"""
        if len(recent_chars) < n - 1:
            return False

        # 构建当前n-gram
        current_gram = recent_chars[-(n-1):] + [token]

        # 检查是否在历史中重复出现
        for i in range(len(recent_chars) - n + 1):
            if recent_chars[i:i+n] == current_gram:
                return True
        return False

    def preprocess_text(self, text):
        """将输入文本转换为模型输入格式"""
        # 字符级分词
        chars = list(text.strip())
        logging.debug(f"Input text tokenized into {len(chars)} characters")

        # 转换为ID
        input_ids = [self.sos_token]
        for char in chars:
            input_ids.append(self.word2id.get(char, self.unk_token))
        input_ids.append(self.eos_token)

        logging.debug(f"Converted to input_ids of length {len(input_ids)}")
        return torch.LongTensor([input_ids]).to(self.device)

    def postprocess_text(self, token_ids):
        """将模型输出的token ID转换回文本"""
        text = []
        for token_id in token_ids:
            if token_id == self.eos_token:
                break
            if token_id in [self.pad_token, self.sos_token]:
                continue
            char = self.id2word.get(token_id, '<UNK>')
            if char not in ['<PAD>', '< SOS >', '<EOS>', '<UNK>']:
                text.append(char)

        result = ''.join(text)
        logging.debug(f"Postprocessed text length: {len(result)}")
        return result

    def chat(self, text):
        try:
            logging.info(f"Processing input text: {text}")

            # 预处理输入文本
            input_ids = self.preprocess_text(text)
            if len(input_ids[0]) <= 2:  # 只有SOS和EOS
                logging.warning("Input text too short")
                return random.choice(self.default_responses)

            # 获取特殊token的ID
            special_tokens = {
                self.word2id.get('<PAD>', 3),
                self.word2id.get('<UNK>', 2),
                self.word2id.get('< SOS >', 0),
                self.word2id.get('<EOS>', 1)
            }

            # 使用模型生成回复
            input_tensor = input_ids
            encoder_outputs, encoder_hidden = self.model.encoder(input_tensor)

            # 初始化解码器输入
            decoder_input = torch.LongTensor(
                [[self.word2id['< SOS >']]]).to(self.device)
            decoder_hidden = encoder_hidden

            # 存储生成的token和它们的概率
            generated_tokens = []
            recent_chars = []  # 用于检测重复
            last_token = None  # 用于检测重复token
            repeat_count = 0  # 用于检测重复次数
            no_repeat_ngram_size = 3  # 防止n-gram重复的大小

            for step in range(self.max_length):
                # 获取解码器输出
                decoder_output, decoder_hidden, _ = self.model.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )

                # 应用temperature
                logits = decoder_output.squeeze(0) / self.temperature

                # 将特殊token的概率设为很小的值
                for special_token in special_tokens:
                    logits[0][special_token] = float('-inf')

                # 防止重复token
                if last_token is not None:
                    logits[0][last_token] = logits[0][last_token] * 0.3  # 更强烈的惩罚

                # 防止n-gram重复
                if len(generated_tokens) >= no_repeat_ngram_size:
                    ngram = generated_tokens[-no_repeat_ngram_size:]
                    for token in ngram:
                        logits[0][token] = logits[0][token] * 0.5

                # 获取top-k个概率最高的token
                probs = F.softmax(logits, dim=-1)
                top_probs, top_indices = probs[0].topk(self.top_k)

                # 选择不会导致重复的token
                selected_token = None
                for token_idx, prob in zip(top_indices, top_probs):
                    token = self.id2word[token_idx.item()]

                    # 检查n-gram重复
                    has_repeat = False
                    for n in range(2, self.ngram_size + 1):
                        if self.check_ngram_repeat(recent_chars, token, n):
                            has_repeat = True
                            break

                    if not has_repeat:
                        selected_token = token_idx.item()
                        break

                # 如果没有找到合适的token，使用概率最高的非特殊token
                if selected_token is None:
                    selected_token = top_indices[0].item()

                # 检查重复token
                if selected_token == last_token:
                    repeat_count += 1
                    if repeat_count > 1:  # 如果连续重复超过1次，强制选择其他token
                        for token_idx in top_indices[1:]:
                            if token_idx.item() != last_token:
                                selected_token = token_idx.item()
                                break
                else:
                    repeat_count = 0

                # 如果达到最小长度且生成了EOS，或者超过最大长度，结束生成
                if len(generated_tokens) >= self.min_length and selected_token == self.eos_token:
                    break

                # 更新状态
                last_token = selected_token
                token_char = self.id2word[selected_token]
                recent_chars.append(token_char)
                if len(recent_chars) > self.ngram_size * 2:
                    recent_chars.pop(0)
                generated_tokens.append(selected_token)

                # 更新decoder_input
                decoder_input = torch.LongTensor(
                    [[selected_token]]).to(self.device)

            # 后处理生成的文本
            response = self.postprocess_text(generated_tokens)

            # 确保回复不为空且长度合适
            if not response or len(response.strip()) < self.min_length:
                logging.warning(
                    "Generated response too short, using default response")
                return random.choice(self.default_responses)

            # 添加标点符号（如果最后没有标点）
            if response[-1] not in '。！？，':
                response += '。'

            # 清理重复的标点符号
            response = re.sub(r'[。！？，]{2,}', lambda m: m.group()[0], response)

            # 如果回复质量不高，使用默认回复
            if len(set(response)) < len(response) * 0.5:  # 如果重复字符过多
                return random.choice(self.default_responses)

            logging.info(f"Generated response: {response}")
            return response.strip()

        except Exception as e:
            logging.error(f"Error in chat method: {str(e)}", exc_info=True)
            return random.choice(self.default_responses)

class ChatModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(ChatModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                             dropout=dropout, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                             dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
    def forward(self, input_ids, target_ids=None, teacher_forcing_ratio=0.5):
        # 检查输入是否为空
        if input_ids is None or input_ids.nelement() == 0:
            return "抱歉，我没有理解您的问题，请重新输入。"
            
        # 检查特殊字符
        special_chars = r'[<>{}[\]|\\^~`]'
        if isinstance(input_ids, str) and re.search(special_chars, input_ids):
            return "抱歉，您的输入包含特殊字符，请重新输入。"
            
        # 检查输入长度
        if isinstance(input_ids, torch.Tensor) and input_ids.size(1) > 50:
            return "抱歉，您的输入过长，请控制在50个字符以内。"
            
        # 正常处理
        batch_size = input_ids.size(0)
        embedded = self.embedding(input_ids)
        
        # 编码器
        encoder_outputs, (hidden, cell) = self.encoder(embedded)
        
        # 解码器
        if target_ids is not None:
            # 训练模式
            target_embedded = self.embedding(target_ids)
            decoder_outputs, _ = self.decoder(target_embedded, (hidden, cell))
            output = self.fc(decoder_outputs)
            return output
        else:
            # 推理模式
            outputs = []
            input = input_ids[:, -1].unsqueeze(1)  # 使用最后一个输入作为解码器的第一个输入
            
            for _ in range(50):  # 最大生成长度
                embedded = self.embedding(input)
                output, (hidden, cell) = self.decoder(embedded, (hidden, cell))
                output = self.fc(output)
                outputs.append(output)
                
                # 使用预测结果作为下一个输入
                top1 = output.argmax(2)
                input = top1
                
            outputs = torch.cat(outputs, dim=1)
            return outputs
