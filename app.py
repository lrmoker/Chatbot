from flask import Flask, render_template, request, jsonify, session
import torch
from model.model import ChatBot
import logging
from datetime import datetime
import re
import time
import json
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)  # 用于session加密

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/web_chatbot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 加载训练数据
def load_train_data():
    try:
        with open('data/train.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 如果数据是列表格式，直接使用
            if isinstance(data, list):
                logging.info(f"成功加载 {len(data)} 条训练数据")
                # 打印前几条数据作为示例
                for i, item in enumerate(data[:3]):
                    logging.info(f"示例数据 {i+1}: {item}")
                return data
            # 如果数据是字典格式，尝试获取conversations字段
            elif isinstance(data, dict):
                conversations = data.get('conversations', [])
                logging.info(f"成功加载 {len(conversations)} 条训练数据")
                # 打印前几条数据作为示例
                for i, item in enumerate(conversations[:3]):
                    logging.info(f"示例数据 {i+1}: {item}")
                return conversations
            else:
                logging.error("训练数据格式不正确")
                return []
    except Exception as e:
        logging.error(f"加载训练数据失败: {str(e)}")
        return []

# 初始化聊天机器人
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    chatbot = ChatBot(
        model_path='checkpoints/best_model.pth',
        vocab_path='checkpoints/vocab.json',
        device=device
    )
    # 加载训练数据
    train_data = load_train_data()
    logging.info(f"聊天机器人初始化成功，使用设备: {device}")
except Exception as e:
    logging.error(f"聊天机器人初始化失败: {str(e)}")
    raise

# 输入验证配置
MAX_INPUT_LENGTH = 100  # 最大输入长度
MAX_PROCESSING_TIME = 10  # 最大处理时间（秒）
INVALID_CHARS_PATTERN = r'[<>{}[\]|\\^~`]'  # 无效字符模式
SUSPICIOUS_PATTERNS = [
    r'(?i)(select|insert|update|delete|drop|alter|exec|union|where|from)',
    r'(?i)(script|javascript|eval|alert|prompt|confirm)',
    r'(?i)(http|ftp|www)',
    r'(?i)(admin|root|password|login)',
]

def validate_input(text):
    """验证输入文本"""
    if not text or not text.strip():
        return False, "输入不能为空"
    
    if len(text) > MAX_INPUT_LENGTH:
        return False, f"输入长度不能超过{MAX_INPUT_LENGTH}个字符"
    
    if re.search(INVALID_CHARS_PATTERN, text):
        return False, "输入包含无效字符"
    
    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, text):
            return False, "输入包含可疑内容"
    
    return True, ""

def find_response_in_train_data(query: str) -> str:
    """从训练数据中查找匹配的回答"""
    logging.info(f"开始查找匹配的回答，查询内容: {query}")
    for i, conv in enumerate(train_data):
        # 检查对话格式
        if isinstance(conv, list) and len(conv) >= 2:
            # 如果是列表格式，第一个元素是输入，第二个元素是输出
            if conv[0] == query:
                logging.info(f"找到匹配的回答 (列表格式): {conv[1]}")
                return conv[1]
        elif isinstance(conv, dict):
            # 如果是字典格式，检查input和output字段
            if conv.get('input') == query:
                logging.info(f"找到匹配的回答 (字典格式): {conv.get('output')}")
                return conv.get('output')
        
        # 每处理1000条数据打印一次进度
        if i % 1000 == 0:
            logging.info(f"已处理 {i} 条数据")
    
    logging.info("未找到匹配的回答")
    return None

@app.route('/')
def home():
    # 初始化会话历史
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        # 获取请求数据
        data = request.get_json()
        if not data:
            return jsonify({
                'code': 400,
                'message': '请求数据不能为空',
                'data': None
            }), 400

        user_message = data.get('message', '')
        
        # 输入验证
        is_valid, error_message = validate_input(user_message)
        if not is_valid:
            return jsonify({
                'code': 400,
                'message': error_message,
                'data': None
            }), 400

        # 记录请求
        logging.info(f"收到聊天请求: {user_message}")
        
        # 获取机器人回复
        start_time = time.time()
        
        # 首先尝试从训练数据中检索
        response = find_response_in_train_data(user_message)
        if response:
            logging.info(f"使用训练数据中的回答: {response}")
        else:
            # 如果没有找到匹配，使用模型生成回答
            logging.info("未找到匹配的训练数据，使用模型生成回答")
            response = chatbot.chat(user_message)
            logging.info(f"模型生成的回答: {response}")
        
        end_time = time.time()
        process_time = end_time - start_time
        
        # 检查处理时间
        if process_time > MAX_PROCESSING_TIME:
            logging.warning(f"处理时间过长: {process_time:.2f}秒")
            return jsonify({
                'code': 408,
                'message': '处理超时，请稍后重试',
                'data': None
            }), 408
        
        # 检查响应质量
        if not response or len(response.strip()) < 5:
            logging.warning("生成的回复质量不佳")
            return jsonify({
                'code': 500,
                'message': '无法生成有效回复，请重试',
                'data': None
            }), 500
        
        # 更新会话历史
        chat_history = session.get('chat_history', [])
        chat_history.append({
            'user': user_message,
            'bot': response,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'process_time': f"{process_time:.2f}秒"
        })
        session['chat_history'] = chat_history
        
        return jsonify({
            'code': 200,
            'message': 'success',
            'data': {
                'response': response,
                'process_time': f"{process_time:.2f}秒"
            }
        })
        
    except Exception as e:
        logging.error(f"处理请求时发生错误: {str(e)}", exc_info=True)
        return jsonify({
            'code': 500,
            'message': f'服务器内部错误: {str(e)}',
            'data': None
        }), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """获取聊天历史"""
    try:
        chat_history = session.get('chat_history', [])
        return jsonify({
            'code': 200,
            'message': 'success',
            'data': chat_history
        })
    except Exception as e:
        return jsonify({
            'code': 500,
            'message': f'获取历史记录失败: {str(e)}',
            'data': None
        }), 500

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    """清除聊天历史"""
    try:
        session['chat_history'] = []
        return jsonify({
            'code': 200,
            'message': '历史记录已清除',
            'data': None
        })
    except Exception as e:
        return jsonify({
            'code': 500,
            'message': f'清除历史记录失败: {str(e)}',
            'data': None
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    try:
        return jsonify({
            'code': 200,
            'message': 'success',
            'data': {
                'status': 'healthy',
                'device': str(device),
                'model_loaded': True,
                'max_input_length': MAX_INPUT_LENGTH,
                'max_processing_time': MAX_PROCESSING_TIME
            }
        })
    except Exception as e:
        return jsonify({
            'code': 500,
            'message': f'服务异常: {str(e)}',
            'data': None
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'code': 404,
        'message': '接口不存在',
        'data': None
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'code': 405,
        'message': '请求方法不允许',
        'data': None
    }), 405

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 