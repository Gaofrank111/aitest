import os
from openai import OpenAI
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

# 加载环境变量
load_dotenv()

app = Flask(__name__)

# 初始化OpenAI客户端
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    timeout=60.0
)

def is_ai_related(question):
    """判断问题是否与AI相关"""
    ai_keywords = {
        # 基础AI术语
        'ai', '人工智能', '智能', 'artificial intelligence',
        
        # AI模型和产品
        'chatgpt', 'gpt', 'claude', 'gemini', 'llama', 'bert', 
        'stable diffusion', 'midjourney', 'dall-e', 'copilot',
        'openai', 'anthropic', 'google bard', '文心一言', '通义千问',
        
        # AI技术领域
        '机器学习', '深度学习', '神经网络', '大语言模型', 
        '计算机视觉', '自然语言处理', '强化学习', 
        'machine learning', 'deep learning', 'neural', 
        'llm', 'nlp', 'cv', 'ml', 'dl',
        
        # AI应用
        '机器人', 'robot', '智能助手', '智能系统', '语音助手',
        '人脸识别', '图像识别', '语音识别', '自动驾驶',
        
        # AI概念和话题
        '算法', '训练', '推理', '数据集', '模型',
        '智能化', '自动化', '特征', '参数', '优化',
        'prompt', 'token', '微调', 'finetune',
        
        # AI伦理和影响
        '人机交互', '人机协作', 'ai伦理', 'ai安全',
        '数字生命', '智能革命', 'agi', '超级智能'
    }
    
    question_lower = question.lower()
    
    # 1. 直接关键词匹配
    if any(keyword in question_lower for keyword in ai_keywords):
        return True
    
    # 2. 检查是否包含AI相关的上下文
    ai_context_patterns = [
        '智能', '机器', '算法', '模型', '学习', '训练',
        '自动', '预测', '识别', '理解', '生成', '优化'
    ]
    
    context_matches = sum(1 for pattern in ai_context_patterns if pattern in question_lower)
    if context_matches >= 2:
        return True
    
    return False

def get_ai_response(question, user_name, user_title):
    """获取AI回答"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个人工智能领域的专家，要用亲切的语气与用户交流，称呼用户为'[姓名][称谓]'。"},
                {"role": "user", "content": f"请记住我是{user_name}{user_title}，我的问题是：{question}"}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"抱歉，{user_name}{user_title}，发生了错误：{str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    user_name = data.get('userName', '')
    user_title = data.get('userTitle', '')
    
    if not question:
        return jsonify({'error': '请输入问题'})
    
    if is_ai_related(question):
        response = get_ai_response(question, user_name, user_title)
    else:
        response = f"对不起，{user_name}{user_title}，我是人工智能小专家，其他问题我不擅长。"
    
    return jsonify({'response': response}) 