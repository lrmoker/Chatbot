import json
import os
from typing import Dict, List, Optional
import torch
from model.model import Seq2Seq
from model.pre_process import Preprocessor
import logging
import re
import time

class Chatbot:
    def __init__(self, model_path: str = "checkpoints/best_model.pth", 
                 vocab_path: str = "checkpoints/vocab.json",
                 data_path: str = "data/train.json"):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 先加载词汇表
        self.preprocessor = Preprocessor()
        # 确保特殊标记存在
        self.preprocessor.word2id = {
            '<PAD>': 0,
            '<BOS>': 1,
            '<EOS>': 2,
            '<UNK>': 3
        }
        self.preprocessor.id2word = {
            0: '<PAD>',
            1: '<BOS>',
            2: '<EOS>',
            3: '<UNK>'
        }
        
        if os.path.exists(self.vocab_path):
            try:
                with open(self.vocab_path, 'r', encoding='utf-8') as f:
                    loaded_vocab = json.load(f)
                    # 保留特殊标记，添加其他词汇
                    for word, idx in loaded_vocab.items():
                        if word not in self.preprocessor.word2id:
                            self.preprocessor.word2id[word] = len(self.preprocessor.word2id)
                            self.preprocessor.id2word[len(self.preprocessor.word2id) - 1] = word
                print("词汇表加载成功")
            except Exception as e:
                print(f"词汇表加载失败: {str(e)}")
                print("将使用随机初始化的词汇表")
        else:
            print("未找到词汇表文件，将使用随机初始化的词汇表")
        
        # 然后加载模型
        self.model = self._load_model()
        
        # 最后加载对话数据用于检索
        self.conversations = self._load_conversations()
    
    def _load_model(self) -> Seq2Seq:
        """加载训练好的模型"""
        vocab_size = len(self.preprocessor.word2id)
        model = Seq2Seq(
            dataClass=self.preprocessor,
            featureSize=256,
            hiddenSize=256,
            numLayers=3,
            dropout=0.1,
            modelType='lstm'
        ).to(self.device)
        
        if os.path.exists(self.model_path):
            try:
                model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                print("模型加载成功")
            except Exception as e:
                print(f"模型加载失败: {str(e)}")
                print("将使用随机初始化的模型")
        else:
            print("未找到模型文件，将使用随机初始化的模型")
        
        model.eval()  # 设置为评估模式
        return model
    
    def _load_conversations(self) -> List[Dict[str, str]]:
        """加载对话数据用于检索"""
        if not os.path.exists(self.data_path):
            print(f"未找到对话数据文件: {self.data_path}")
            return []
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                conversations = data.get('conversations', [])
                print(f"成功加载 {len(conversations)} 条对话数据")
                return conversations
        except Exception as e:
            print(f"加载对话数据失败: {str(e)}")
            return []
    
    def find_response(self, query: str) -> Optional[str]:
        """首先尝试从对话数据中检索匹配的回答"""
        for conv in self.conversations:
            if conv['input'] == query:
                return conv['output']
        return None
    
    def generate_response(self, query: str) -> str:
        """使用模型生成回答"""
        try:
            # 将输入转换为模型可处理的格式
            input_ids = [self.preprocessor.word2id['<BOS>']]
            input_ids.extend(self.preprocessor.text_to_ids(query))
            input_ids.append(self.preprocessor.word2id['<EOS>'])
            
            input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
            input_length = torch.tensor([len(input_ids)], dtype=torch.long).to(self.device)
            
            # 生成回答
            with torch.no_grad():
                output_ids = self.model.predictByBeamSearch(
                    input_seq=input_tensor,
                    input_length=input_length,
                    maxAnswerLength=20,
                    beamWidth=3
                )
                response = self.preprocessor.ids_to_text(output_ids)
            
            return response
        except Exception as e:
            print(f"生成回答时出错: {str(e)}")
            return "抱歉，我无法生成回答。"
    
    def chat(self):
        """开始对话"""
        print("欢迎使用聊天机器人！输入'退出'结束对话。")
        
        while True:
            user_input = input("\n用户: ").strip()
            if user_input.lower() in ['退出', 'exit', 'quit']:
                print("再见！")
                break
            
            # 检查输入是否有效
            is_valid, error_msg = check_input(user_input)
            if not is_valid:
                print(error_msg)
                continue
            
            # 记录开始时间
            start_time = time.time()
            
            # 首先尝试从对话数据中检索
            response = self.find_response(user_input)
            if response:
                # 计算响应时间
                response_time = time.time() - start_time
                print(f"机器人: {response} (响应时间: {response_time:.2f}秒)")
            else:
                # 如果没有找到匹配，使用模型生成回答
                response = self.generate_response(user_input)
                # 计算响应时间
                response_time = time.time() - start_time
                print(f"机器人: {response} (响应时间: {response_time:.2f}秒)")

def check_input(text):
    """检查输入是否有效"""
    # 检查空输入
    if not text or not text.strip():
        return False, "抱歉，我没有理解您的问题，请重新输入。"
        
    # 检查特殊字符
    special_chars = r'[<>{}[\]|\\^~`]'
    if re.search(special_chars, text):
        return False, "抱歉，您的输入包含特殊字符，请重新输入。"
        
    # 检查输入长度
    if len(text) > 50:
        return False, "抱歉，您的输入过长，请控制在50个字符以内。"
        
    return True, None

def main():
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 加载模型和词汇表
    model_path = "model/chatbot_model.pth"
    vocab_path = "model/vocab.json"
    
    try:
        chatbot = Chatbot(model_path, vocab_path)
        logging.info("模型加载成功")
        
        while True:
            # 获取用户输入
            user_input = input("\n请输入您的问题（输入'quit'退出）: ")
            
            # 检查是否退出
            if user_input.lower() == 'quit':
                break
                
            # 检查输入是否有效
            is_valid, error_msg = check_input(user_input)
            if not is_valid:
                print(error_msg)
                continue
                
            # 获取回复
            response = chatbot.chat()
            print(f"\n回复: {response}")
            
    except Exception as e:
        logging.error(f"发生错误: {str(e)}")
        print("抱歉，系统出现错误，请稍后再试。")

if __name__ == "__main__":
    main() 