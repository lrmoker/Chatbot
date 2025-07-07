# 导入库
from model.model import ChatBot
import logging
import os
import torch
import json
import time

def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载词汇表
    vocab = load_vocab('checkpoints/vocab.json')
    print(f"词汇表大小: {len(vocab)}")
    
    # 初始化聊天机器人
    print("正在初始化聊天机器人...")
    chatbot = ChatBot(
        model_path='checkpoints/best_model.pth',
        vocab_path='checkpoints/vocab.json',
        device=device
    )
    print("初始化完成！\n")
    
    # 开始对话
    print("开始对话（输入 'exit' 结束对话）：")
    while True:
        try:
            # 获取用户输入
            user_input = input("\n用户: ").strip()
            
            # 检查是否退出
            if user_input.lower() == 'exit':
                print("再见！")
                break
            
            if not user_input:
                print("请输入有效的文本！")
                continue
            
            # 记录开始时间
            start_time = time.time()
            
            # 获取回复
            response = chatbot.chat(user_input)
            
            # 计算响应时间
            elapsed_time = time.time() - start_time
            
            # 打印回复和响应时间
            print(f"伊兹: {response}")
            print(f"响应时间: {elapsed_time:.2f}秒")
            
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"发生错误: {str(e)}")
            continue

if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # 输出到控制台
            logging.FileHandler('data/chatbot.log')  # 更新日志文件路径
        ]
    )
    
    # 运行测试
    main()
