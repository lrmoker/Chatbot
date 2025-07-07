# 导入库
from model.model import ChatBot
import torch
import warnings
import argparse
from torch.serialization import add_safe_globals
import time
import random

warnings.filterwarnings("ignore")

# 添加安全的全局变量
add_safe_globals([])

# 预定义一些人格特征和回复模板
PERSONALITY_TRAITS = [
    "我是一个友好的AI助手",
    "我喜欢和人聊天",
    "我会认真倾听",
    "我会尽力帮助他人",
    "我希望能成为你的好朋友",
    "我在不断学习和进步"
]

GREETING_TEMPLATES = [
    "你好啊！今天感觉如何？",
    "很高兴见到你！",
    "你好，有什么我可以帮你的吗？",
    "嗨，希望你今天过得愉快！",
    "你好，让我们来聊聊天吧！",
    "见到你真好！想聊些什么呢？"
]

FALLBACK_RESPONSES = [
    "这个问题很有趣，让我想想...",
    "抱歉，我可能需要更多信息...",
    "这个话题很有意思，能详细说说吗？",
    "我明白你的意思了，不过...",
    "嗯，这样啊...",
    "让我理解一下你的意思...",
    "说得对，继续说说看...",
    "这个想法很棒！",
    "确实如此呢",
    "有道理，我觉得..."
]

POSITIVE_RESPONSES = [
    "真好啊！",
    "太棒了！",
    "我也这么觉得！",
    "说得对！",
    "很高兴听到这个！"
]

class DialogueManager:
    def __init__(self, max_history=5):
        self.history = []
        self.max_history = max_history
        self.last_response_time = time.time()
        self.consecutive_short_responses = 0
        
    def add_to_history(self, user_input, bot_response):
        self.history.append((user_input, bot_response))
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
    def get_context(self):
        # 只使用最近的对话历史
        recent_history = self.history[-3:] if len(self.history) > 3 else self.history
        return " ".join([f"{u}" for u, _ in recent_history])
    
    def enhance_response(self, response, user_input):
        # 如果响应为空或太短，使用模型重新生成
        if not response or len(response) < 2:
            self.consecutive_short_responses += 1
            if self.consecutive_short_responses > 2:
                # 连续短回复时，重新组织输入以获得更好的回答
                context = self.get_context()
                return f"让我重新思考一下你说的 '{user_input}'"
            return response
        
        self.consecutive_short_responses = 0
        return response

    def process_user_input(self, input_text):
        # 移除简单的预定义回复，让模型处理所有回答
        return None

# 选择哪个模型，是否使用gpu
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='The path of your model file.', required=True, type=str)
parser.add_argument('--device', help='Your program running environment, "cpu" or "cuda"', type=str, default='cpu')
args = parser.parse_args()

# 确保device有效
if args.device not in ['cpu', 'cuda']:
    print("无效的设备类型，请选择 'cpu' 或 'cuda'")
    exit()

if args.device == 'cuda' and not torch.cuda.is_available():
    print("警告：CUDA不可用，自动切换到CPU模式")
    args.device = 'cpu'

print(args)

# 主程序，打印log，同时对话
if __name__ == "__main__":
    print('加载模型中...')
    device = torch.device(args.device)
    chatBot = ChatBot(
        model_path=args.model,
        model_type='cnn'  # 使用CNN模型
    )
    dialogue_manager = DialogueManager()
    print('模型加载完成...')
    
    # 更个性化的欢迎语
    print('伊兹: 你好！我是伊兹，一个AI助手。让我们开始对话吧！')

    allRandomChoose, showInfo = False, False

    while True:
        try:
            inputSeq = input("主人: ").strip()
            
            if not inputSeq:
                print('伊兹: 你想聊些什么呢？我在认真听。')
                continue

            if inputSeq == '_crazy_on_':
                allRandomChoose = True
                print('伊兹: 已切换到创意模式，我会更有想象力地回答。')
            elif inputSeq == '_crazy_off_':
                allRandomChoose = False
                print('伊兹: 已切换到标准模式，我会更严谨地回答。')
            elif inputSeq == '_showInfo_on_':
                showInfo = True
                print('伊兹: 已开启详细模式，我会解释我的思考过程。')
            elif inputSeq == '_showInfo_off_':
                showInfo = False
                print('伊兹: 已关闭详细模式。')
            elif inputSeq.lower() == 'exit':
                print('伊兹: 感谢与你的对话！下次再见。')
                break
            else:
                try:
                    # 获取对话历史上下文
                    context = dialogue_manager.get_context()
                    
                    # 构建增强的输入，包含上下文信息
                    if context and len(context.strip()) > 0:
                        enhanced_input = f"{context} {inputSeq}"
                    else:
                        enhanced_input = inputSeq
                    
                    # 使用模型生成回复
                    outputSeq = chatBot.chat(enhanced_input)
                    
                    # 增强回复
                    if outputSeq and len(outputSeq.strip()) > 0:
                        enhanced_response = dialogue_manager.enhance_response(outputSeq, inputSeq)
                    else:
                        enhanced_response = "抱歉，我需要重新思考一下这个问题。"
                    
                    # 添加到历史记录
                    dialogue_manager.add_to_history(inputSeq, enhanced_response)
                    
                    print('伊兹: ', enhanced_response)
                    
                    if showInfo:
                        print(f"(Debug: 使用的上下文: {context})")
                except Exception as e:
                    print('伊兹: 抱歉，我需要重新思考一下这个问题。你能换个方式问我吗？')
                    if showInfo:
                        print(f"(Debug: {str(e)})")
            print()
        except KeyboardInterrupt:
            print('\n伊兹: 感谢与你的对话！下次再见。')
            break
        except Exception as e:
            print(f'发生错误: {str(e)}')
            continue
