document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    // 自动调整文本框高度
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });

    // 发送消息函数
    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        // 添加用户消息到聊天界面
        addMessage(message, 'user');
        
        // 清空输入框
        userInput.value = '';
        userInput.style.height = 'auto';
        
        // 禁用发送按钮
        sendButton.disabled = true;

        try {
            // 发送请求到服务器
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });

            const data = await response.json();

            if (response.ok) {
                // 添加机器人回复到聊天界面
                addMessage(data.response, 'bot');
            } else {
                // 显示错误消息
                addMessage('抱歉，发生了一些错误，请稍后再试。', 'bot');
            }
        } catch (error) {
            console.error('Error:', error);
            addMessage('抱歉，网络连接出现问题，请稍后再试。', 'bot');
        } finally {
            // 重新启用发送按钮
            sendButton.disabled = false;
        }
    }

    // 添加消息到聊天界面
    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = text;
        
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        
        // 滚动到底部
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // 发送按钮点击事件
    sendButton.addEventListener('click', sendMessage);

    // 回车发送消息
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
}); 