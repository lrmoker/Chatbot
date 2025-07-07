# 导入库 正则 分词
import re, jieba, random, time
import numpy as np
import torch
from sklearn.model_selection import train_test_split  # 数据集的划分
from collections import Counter
import json
import csv
import pandas as pd
import os
from tqdm import tqdm
import logging
from typing import List, Tuple


# 定义语料的类
class Corpus:
    """语料库类，用于加载和处理对话数据"""
    def __init__(self, filePath=None, max_sentence_word_count=50):
        self.filePath = filePath
        self.max_sentence_word_count = max_sentence_word_count
        self.word2id = None
        self.id2word = None
        self.wordNum = 0
        self.chatDataWord = []
        self.chatDataId = []
        self.QChatDataId = []
        self.AChatDataId = []
        self.QLens = None
        self.ALens = None
        self.QMaxLen = 0
        self.AMaxLen = 0
        self.trainIdList = []
        self.testIdList = []
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }

    def load_from_json(self, file_path: str) -> List[Tuple[List[str], List[str]]]:
        """
        从JSON文件加载对话数据
        
        Args:
            file_path (str): JSON文件路径
            
        Returns:
            List[Tuple[List[str], List[str]]]: 处理后的问答对列表
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValueError("JSON data must be a list of conversations")
            
            processed_data = []
            for conv in tqdm(data, desc="Processing conversations"):
                if not isinstance(conv, dict) or 'conversations' not in conv:
                    continue
                    
                for i in range(0, len(conv['conversations'])-1, 2):
                    if i+1 >= len(conv['conversations']):
                        break
                        
                    question = conv['conversations'][i]['value']
                    answer = conv['conversations'][i+1]['value']
                    
                    if not self._is_valid_qa_pair(question, answer):
                        continue
                    
                    q_tokens = self._segment_text(question)
                    a_tokens = self._segment_text(answer)
                    
                    if q_tokens and a_tokens:
                        processed_data.append([q_tokens, a_tokens])
            
            if not processed_data:
                raise ValueError("No valid conversations found in the data")
                
            logging.info(f"Successfully loaded {len(processed_data)} valid conversations")
            return processed_data
            
        except Exception as e:
            logging.error(f"Error loading JSON file: {str(e)}")
            raise

    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing unwanted characters and patterns
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove URLs (simple version)
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Keep only Chinese characters, letters, numbers, and basic punctuation
        text = re.sub(r'[^\u4e00-\u9fff\w\s.,?!，。？！、]', '', text)
        
        return text.strip()

    def _is_valid_qa_pair(self, question: str, answer: str) -> bool:
        """
        Check if a question-answer pair is valid
        
        Args:
            question (str): The question text
            answer (str): The answer text
            
        Returns:
            bool: True if the pair is valid, False otherwise
        """
        if not question or not answer:
            return False
        if len(question) < 2 or len(answer) < 2:
            return False
        if len(question) > 50 or len(answer) > 50:
            return False
        return True

    def _QALens(self, chatDataId: List[List[int]]) -> None:
        """
        Calculate lengths of questions and answers
        
        Args:
            chatDataId (List[List[int]]): List of question-answer pairs as IDs
        """
        try:
            self.QLens = np.array([len(qa[0]) for qa in chatDataId], dtype='int32')
            self.ALens = np.array([len(qa[1]) for qa in chatDataId], dtype='int32')
            self.QMaxLen = max(self.QLens)
            self.AMaxLen = max(self.ALens)
            
            logging.info(f"Max question length: {self.QMaxLen}")
            logging.info(f"Max answer length: {self.AMaxLen}")
            logging.info(f"Average question length: {np.mean(self.QLens):.2f}")
            logging.info(f"Average answer length: {np.mean(self.ALens):.2f}")
        except Exception as e:
            logging.error(f"Error calculating QA lengths: {str(e)}")
            raise

    def _word_id_map(self, data: List[List[List[str]]]) -> None:
        """
        Build word-to-id mapping from the data
        
        Args:
            data (List[List[List[str]]]): List of question-answer pairs as tokens
        """
        try:
            if self.word2id is None:
                # Count word frequencies
                word_counts = Counter()
                for qa in tqdm(data, desc="Counting words"):
                    word_counts.update(qa[0])  # Question words
                    word_counts.update(qa[1])  # Answer words
                
                # Sort words by frequency
                sorted_words = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
                
                # Initialize word-to-id mapping with special tokens
                self.word2id = dict(self.special_tokens)
                
                # Add remaining words
                for word, _ in sorted_words:
                    if word not in self.word2id:
                        self.word2id[word] = len(self.word2id)
                
                # Create id-to-word mapping
                self.id2word = {v: k for k, v in self.word2id.items()}
                self.wordNum = len(self.word2id)
                
                logging.info(f"Vocabulary size: {self.wordNum}")
                logging.info(f"Most common words: {sorted_words[:10]}")
                logging.info(f"Special tokens: {list(self.special_tokens.keys())}")
        except Exception as e:
            logging.error(f"Error building word ID map: {str(e)}")
            raise

    def _dataEnhance(self, samples: List[int], ratio: float, eosToken: int, unkToken: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform data augmentation by randomly replacing words with UNK token
        
        Args:
            samples (List[int]): List of sample indices
            ratio (float): Probability of applying augmentation
            eosToken (int): End of sequence token ID
            unkToken (int): Unknown token ID
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Augmented data
        """
        try:
            QMaxLen = max(self.QLens[samples])
            AMaxLen = max(self.ALens[samples])
            
            QDataId = []
            ADataId = []
            
            for i in samples:
                # Process question
                q = self.QChatDataId[i].copy()
                if random.random() < ratio:
                    # Randomly replace some words with UNK
                    for j in range(len(q)):
                        if random.random() < 0.1:  # 10% chance to replace with UNK
                            q[j] = unkToken
                QDataId.append(q + [eosToken] * (QMaxLen - len(q) + 1))
                
                # Process answer
                a = self.AChatDataId[i].copy()
                if random.random() < ratio:
                    # Randomly replace some words with UNK
                    for j in range(len(a)):
                        if random.random() < 0.1:  # 10% chance to replace with UNK
                            a[j] = unkToken
                ADataId.append(a + [eosToken] * (AMaxLen - len(a) + 1))
            
            return (np.array(QDataId, dtype='int32'), 
                    self.QLens[samples],
                    np.array(ADataId, dtype='int32'), 
                    self.ALens[samples])
        except Exception as e:
            logging.error(f"Error in data enhancement: {str(e)}")
            raise

    def load_from_tsv(self, filePath):
        """从TSV文件加载数据（保持原有功能）"""
        # ... 保持原有的TSV加载代码不变 ...
        pass

    def words_to_ids(self, words):
        """将词列表转换为ID序列"""
        # 添加开始和结束标记
        return [self.word2id["< SOS >"]] + [self.word2id.get(word, self.word2id["<UNK>"]) for word in words] + [self.word2id["<EOS>"]]

    def random_batch_data_stream(self, batchSize=10, isEval=False):
        """生成随机批次数据流
        Args:
            batchSize: 批次大小
            isEval: 是否用于评估（使用验证集）
        """
        samples = self.validationSamples if isEval else self.trainingSamples
        sampleNum = len(samples)
        
        # 创建索引列表并打乱
        indices = list(range(sampleNum))
        random.shuffle(indices)
        
        # 生成批次数据
        for start in range(0, sampleNum, batchSize):
            end = min(start + batchSize, sampleNum)
            batch_indices = indices[start:end]
            
            # 获取这个批次的最大长度
            max_q_len = max(len(samples[i][0]) for i in batch_indices)
            max_a_len = max(len(samples[i][1]) for i in batch_indices)
            
            # 创建批次张量
            batch_q = np.zeros([end - start, max_q_len], dtype=np.int32)
            batch_a = np.zeros([end - start, max_a_len], dtype=np.int32)
            batch_q_lens = []
            batch_a_lens = []
            
            # 填充数据
            for i, idx in enumerate(batch_indices):
                q_ids, a_ids = samples[idx]
                q_len, a_len = len(q_ids), len(a_ids)
                
                batch_q[i, :q_len] = q_ids
                batch_a[i, :a_len] = a_ids
                batch_q_lens.append(q_len)
                batch_a_lens.append(a_len)
            
            # 转换为PyTorch张量
            batch_q = torch.tensor(batch_q, dtype=torch.long)
            batch_a = torch.tensor(batch_a, dtype=torch.long)
            batch_q_lens = torch.tensor(batch_q_lens, dtype=torch.int)
            batch_a_lens = torch.tensor(batch_a_lens, dtype=torch.int)
            
            yield batch_q, batch_q_lens, batch_a, batch_a_lens

    def _segment_text(self, text):
        """
        对文本进行分词
        :param text: 待分词文本
        :return: 分词后的文本列表
        """
        try:
            # 使用正则表达式预处理文本
            text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？]', '', text)
            
            # 对于英文和数字，保持完整
            words = []
            i = 0
            while i < len(text):
                if text[i].isascii():  # 英文或数字
                    start = i
                    while i < len(text) and text[i].isascii():
                        i += 1
                    words.append(text[start:i])
                else:  # 中文字符
                    words.append(text[i])
                    i += 1
            
            return words
        except Exception as e:
            logging.error(f"分词过程出错: {str(e)}")
            return list(text)  # 如果分词失败，返回单字符列表

    def _load_data(self, filePath):
        """
        加载数据文件
        :param filePath: 文件路径
        :return: 处理后的问答对列表
        """
        def process_lines(lines):
            processed_data = []
            total_lines = len(lines)
            logging.info(f"开始处理数据文件，共 {total_lines} 行")
            
            for i, line in enumerate(tqdm(lines, desc="处理数据")):
                try:
                    if not line.strip():
                        continue
                        
                    parts = line.strip().split('\t')
                    if len(parts) != 2:
                        continue
                        
                    question, answer = parts
                    
                    # 检查问答对的有效性
                    if not self._is_valid_qa_pair(question, answer):
                        continue
                    
                    q_tokens = self._segment_text(question)
                    a_tokens = self._segment_text(answer)
                    
                    if q_tokens and a_tokens:  # 确保分词结果非空
                        processed_data.append([q_tokens, a_tokens])
                        
                except Exception as e:
                    logging.warning(f"处理第 {i+1} 行时出错: {str(e)}")
                    continue
            
            return processed_data
        
        # 首先尝试UTF-8编码
        try:
            with open(filePath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            processed_data = process_lines(lines)
            
        except UnicodeDecodeError:
            logging.warning("UTF-8编码打开失败，尝试使用GBK编码")
            try:
                with open(filePath, 'r', encoding='gbk') as f:
                    lines = f.readlines()
                processed_data = process_lines(lines)
                
            except Exception as e:
                logging.error(f"加载数据文件失败: {str(e)}")
                raise
        
        if not processed_data:
            raise ValueError("没有找到有效的问答对")
            
        logging.info(f"成功加载 {len(processed_data)} 个有效问答对")
        return processed_data

    # 重置词的 ID 和映射关系
    def reset_word_id_map(self, id2word, word2id):
        self.id2word, self.word2id = id2word, word2id
        chatDataId = [[[self.word2id[w] for w in qa[0]], [self.word2id[w] for w in qa[1]]] for qa in self.chatDataWord]
        self.QChatDataId, self.AChatDataId = [qa[0] for qa in chatDataId], [qa[1] for qa in chatDataId]

    # 确定一个 epoch 数据流 确认数据是训练数据
    def one_epoch_data_stream(self, batchSize=16, isDataEnhance=False, dataEnhanceRatio=0.2, type='train'):
        """生成一个epoch的数据流"""
        idList = self.trainIdList if type == 'train' else self.testIdList
        eosToken = self.word2id['<EOS>']
        unkToken = self.word2id['<UNK>']
        
        for i in range((len(idList) + batchSize - 1) // batchSize):
            samples = idList[i * batchSize:(i + 1) * batchSize]
            
            if isDataEnhance:
                yield self._dataEnhance(samples, dataEnhanceRatio, eosToken, unkToken)
            else:
                QMaxLen = max(self.QLens[samples])
                AMaxLen = max(self.ALens[samples])
                
                QDataId = np.array([
                    self.QChatDataId[i] + [eosToken] * (QMaxLen - self.QLens[i] + 1)
                    for i in samples
                ], dtype='int32')
                
                ADataId = np.array([
                    self.AChatDataId[i] + [eosToken] * (AMaxLen - self.ALens[i] + 1)
                    for i in samples
                ], dtype='int32')
                
                yield QDataId, self.QLens[samples], ADataId, self.ALens[samples]

    # 遍历
    # 清洗数据：去除低质量内容
    def _purify(self, txt):
        """清洗数据：去除低质量内容"""
        clean_data = []
        for line in txt:
            # 规范化文本
            line = self._normalize_text(line)
            
            # 分割问答对
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            
            question, answer = parts
            question = self._normalize_text(question)
            answer = self._normalize_text(answer)
            
            # 过滤无效对话
            if not self._is_valid_qa_pair(question, answer):
                continue
            
            clean_data.append(f"{question}\t{answer}")
        
        return clean_data

    def _normalize_text(self, text):
        """规范化文本，处理编码和特殊字符"""
        if not isinstance(text, str):
            text = str(text)
        
        # 尝试不同的编码转换
        try:
            text = text.encode('utf-8').decode('utf-8')
        except UnicodeError:
            try:
                text = text.encode('gbk').decode('gbk')
            except UnicodeError:
                try:
                    text = text.encode('gb2312').decode('gb2312')
                except UnicodeError:
                    text = text.encode('gb18030').decode('gb18030')
        
        # 替换特殊字符
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = text.replace('\t', ' ')
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

    # 统计词频并构建词表
    def _word_id_map(self, data):
        """Build word-to-id mapping"""
        if self.word2id is None:
            # Count word frequencies
            word_counts = Counter()
            for qa in tqdm(data, desc="Counting words"):
                word_counts.update(qa[0])  # Question words
                word_counts.update(qa[1])  # Answer words
            
            # Sort words by frequency
            sorted_words = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
            
            # Initialize word-to-id mapping with special tokens
            self.word2id = dict(self.special_tokens)
            
            # Add remaining words
            for word, _ in sorted_words:
                if word not in self.word2id:
                    self.word2id[word] = len(self.word2id)
            
            # Create id-to-word mapping
            self.id2word = {v: k for k, v in self.word2id.items()}
            self.wordNum = len(self.word2id)
            
            print(f"Vocabulary size: {self.wordNum}")
            print(f"Most common words: {sorted_words[:10]}")
            print(f"Special tokens: {list(self.special_tokens.keys())}")


# seq2id 函数
def seq2id(word2id, seqData):
    seqId = [word2id[w] for w in seqData]
    return seqId


def seq2id(word2id, seqData):
    seqId = [word2id[w] for w in seqData]
    return seqId

def id2seq(id2word, seqId):
    seqData = [id2word[i] for i in seqId]
    return seqData

#去掉一些停用词
def filter_sent(sent):
    return sent.replace('\n','').replace(' ','').replace('，',',').replace('。','.').replace('；',';').replace('：',':').replace('？','?').replace('！','!').replace('"','"').replace('"','"').replace("'","'").replace("'","'").replace('(','(').replace(')',')')

class Preprocessor:
    """文本预处理类，用于处理输入文本"""
    def __init__(self, max_length=50):
        self.max_length = max_length
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }
        self.word2id = dict(self.special_tokens)
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.vocab = set(self.word2id.keys())
        self.vocab_size = len(self.vocab)

    def build_vocab(self, texts: List[str], min_freq: int = 2) -> None:
        """
        从文本构建词汇表
        
        Args:
            texts (List[str]): 文本列表
            min_freq (int): 最小词频，默认为2
        """
        try:
            # 统计词频
            word_counts = Counter()
            for text in tqdm(texts, desc="Building vocabulary"):
                words = self.segment_text(text)
                word_counts.update(words)
            
            # 添加高频词到词汇表
            for word, count in word_counts.items():
                if count >= min_freq and word not in self.word2id:
                    self.word2id[word] = len(self.word2id)
                    self.id2word[len(self.id2word)] = word
            
            # 更新词汇表
            self.vocab = set(self.word2id.keys())
            self.vocab_size = len(self.vocab)
            
            logging.info(f"Vocabulary size: {self.vocab_size}")
            logging.info(f"Most common words: {word_counts.most_common(10)}")
            
        except Exception as e:
            logging.error(f"Error building vocabulary: {str(e)}")
            raise

    def text_to_ids(self, text: str) -> List[int]:
        """
        将文本转换为ID序列
        
        Args:
            text (str): 输入文本
            
        Returns:
            List[int]: ID序列
        """
        try:
            words = self.segment_text(text)
            ids = [self.word2id.get(word, self.word2id['<UNK>']) for word in words]
            return [self.word2id['<BOS>']] + ids + [self.word2id['<EOS>']]
        except Exception as e:
            logging.error(f"Error converting text to IDs: {str(e)}")
            return [self.word2id['<UNK>']]

    def ids_to_text(self, ids: List[int]) -> str:
        """
        将ID序列转换为文本
        
        Args:
            ids (List[int]): ID序列
            
        Returns:
            str: 转换后的文本
        """
        try:
            words = [self.id2word.get(id, '<UNK>') for id in ids]
            # 移除特殊标记
            words = [w for w in words if w not in ['<BOS>', '<EOS>', '<PAD>']]
            return ''.join(words)
        except Exception as e:
            logging.error(f"Error converting IDs to text: {str(e)}")
            return '<UNK>'

    def clean_text(self, text: str) -> str:
        """
        清理文本，去除不需要的字符和模式
        
        Args:
            text (str): 输入文本
            
        Returns:
            str: 清理后的文本
        """
        if not isinstance(text, str):
            return ""
            
        # 移除多余空白
        text = ' '.join(text.split())
        
        # 移除URL（简单版本）
        text = re.sub(r'https?://\S+', '', text)
        
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 只保留中文字符、字母、数字和基本标点
        text = re.sub(r'[^\u4e00-\u9fff\w\s.,?!，。？！、]', '', text)
        
        return text.strip()

    def segment_text(self, text: str) -> List[str]:
        """
        对文本进行分词
        
        Args:
            text (str): 输入文本
            
        Returns:
            List[str]: 分词后的文本列表
        """
        try:
            # 使用正则表达式预处理文本
            text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？]', '', text)
            
            # 对于英文和数字，保持完整
            words = []
            i = 0
            while i < len(text):
                if text[i].isascii():  # 英文或数字
                    start = i
                    while i < len(text) and text[i].isascii():
                        i += 1
                    words.append(text[start:i])
                else:  # 中文字符
                    words.append(text[i])
                    i += 1
            
            return words
        except Exception as e:
            logging.error(f"分词过程出错: {str(e)}")
            return list(text)  # 如果分词失败，返回单字符列表

    def normalize_text(self, text: str) -> str:
        """
        规范化文本，处理编码和特殊字符
        
        Args:
            text (str): 输入文本
            
        Returns:
            str: 规范化后的文本
        """
        if not isinstance(text, str):
            text = str(text)
        
        # 尝试不同的编码转换
        try:
            text = text.encode('utf-8').decode('utf-8')
        except UnicodeError:
            try:
                text = text.encode('gbk').decode('gbk')
            except UnicodeError:
                try:
                    text = text.encode('gb2312').decode('gb2312')
                except UnicodeError:
                    text = text.encode('gb18030').decode('gb18030')
        
        # 替换特殊字符
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = text.replace('\t', ' ')
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

    def is_valid_text(self, text: str) -> bool:
        """
        检查文本是否有效
        
        Args:
            text (str): 输入文本
            
        Returns:
            bool: 如果文本有效返回True，否则返回False
        """
        if not text:
            return False
        if len(text) < 2:
            return False
        if len(text) > self.max_length:
            return False
        return True

    def filter_sent(self, sent: str) -> str:
        """
        过滤句子中的停用词和特殊字符
        
        Args:
            sent (str): 输入句子
            
        Returns:
            str: 过滤后的句子
        """
        return sent.replace('\n','').replace(' ','').replace('，',',').replace('。','.').replace('；',';').replace('：',':').replace('？','?').replace('！','!').replace('"','"').replace('"','"').replace("'","'").replace("'","'").replace('(','(').replace(')',')')