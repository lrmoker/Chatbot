import json
import random
import os
from tqdm import tqdm  # 添加进度条支持

class DatasetGenerator:
    def __init__(self):
        # 扩展问题模板
        self.question_templates = [
            "你能告诉我{topic}吗？",
            "请问{topic}是什么？",
            "我想了解{topic}",
            "{topic}是怎么回事？",
            "能详细说说{topic}吗？",
            "关于{topic}你怎么看？",
            "{topic}有什么特点？",
            "为什么{topic}会这样？",
            "{topic}的原因是什么？",
            "如何看待{topic}？",
            "谈谈你对{topic}的看法",
            "{topic}的发展趋势如何？",
            "你对{topic}有什么建议？",
            "{topic}存在哪些问题？",
            "怎样才能更好地理解{topic}？",
            "{topic}给我们带来了什么影响？",
            "请介绍一下{topic}",
            "{topic}的未来会怎样？",
            "{topic}有哪些优点和缺点？",
            "分析一下{topic}的现状"
        ]

        # 扩展主题和对应的回答
        self.topics = {
            "人工智能": [
                "人工智能是一门研究如何让计算机模拟人类智能的科学。它包括机器学习、深度学习等多个领域。",
                "人工智能技术可以帮助我们解决复杂问题，提高工作效率，创造新的可能性。",
                "现代人工智能主要基于深度学习和大数据，通过算法来学习和理解世界。",
                "人工智能正在各个领域发挥重要作用，从医疗诊断到自动驾驶都有其应用。",
                "人工智能的发展需要考虑伦理和安全问题，确保技术发展造福人类。"
            ],
            "环境保护": [
                "环境保护是为了保护地球生态系统，维护人类和其他生物的生存环境。",
                "我们可以通过节约能源、减少污染、保护野生动物等方式来保护环境。",
                "环境保护需要每个人的参与，从日常生活做起，共同维护地球家园。",
                "气候变化是当前最严峻的环境问题之一，需要全球共同努力应对。",
                "可持续发展是环境保护的重要理念，要平衡发展和环境保护的关系。"
            ],
            "健康生活": [
                "健康生活需要均衡饮食、规律作息和适度运动。",
                "保持良好的生活习惯，注意饮食营养均衡，适当运动是健康的基础。",
                "健康的生活方式包括充足的睡眠、营养均衡的饮食和定期运动。",
                "心理健康同样重要，要学会调节心情，保持积极乐观的心态。",
                "预防胜于治疗，养成健康的生活习惯是最好的保健方式。"
            ],
            "教育": [
                "教育是提高人的素质和能力的重要途径，对个人发展和社会进步都很重要。",
                "好的教育应该注重培养学生的创造力和独立思考能力。",
                "教育不仅是传授知识，更重要的是培养人的全面发展。",
                "现代教育需要与时俱进，融入新技术和新理念。",
                "终身学习是现代社会的必然趋势，要持续更新知识和技能。"
            ],
            "科技发展": [
                "科技发展日新月异，给我们的生活带来了巨大的变化和便利。",
                "现代科技的发展极大地提高了人类的生活质量和工作效率。",
                "科技创新推动着人类社会不断进步，解决各种挑战。",
                "新技术的应用需要考虑社会影响和伦理问题。",
                "科技发展要以人为本，服务于人类社会的进步。"
            ],
            "经济发展": [
                "经济发展需要创新驱动，转变发展方式，提高质量和效益。",
                "可持续的经济发展要平衡效益、社会责任和环境保护。",
                "数字经济正在成为经济增长的新引擎，带来新的机遇和挑战。",
                "经济全球化使各国经济相互依存，需要加强国际合作。",
                "普惠金融和共同富裕是经济发展的重要目标。"
            ],
            "文化传承": [
                "文化传承是维系民族特色和历史记忆的重要纽带。",
                "传统文化需要创新发展，与现代生活相融合。",
                "文化多样性是人类文明的重要特征，需要互相尊重和交流。",
                "保护非物质文化遗产对传承文化有重要意义。",
                "文化自信是民族发展的重要精神力量。"
            ],
            "社会发展": [
                "社会发展要以人民为中心，增进民生福祉。",
                "社会治理需要政府、市场、社会多方协同。",
                "建设和谐社会需要法治保障和道德支撑。",
                "社会公平正义是永恒的追求，需要不断完善制度。",
                "社会进步要兼顾效率和公平，促进共同发展。"
            ]
        }

        # 扩展情感和语气
        self.emotions = {
            "积极": ["很高兴", "非常好", "很棒", "真不错", "太好了", "令人振奋", "特别欣慰", "让人期待"],
            "中性": ["我认为", "我觉得", "据我所知", "一般来说", "通常", "客观地说", "从整体看", "综合来看"],
            "思考": ["让我想想", "这个问题很有趣", "从这个角度看", "仔细分析", "深入思考", "值得探讨", "需要考虑", "可以这样理解"],
            "专业": ["从专业角度", "根据研究", "数据表明", "实践证明", "理论分析", "经验显示", "调查发现"]
        }

        # 扩展连接词
        self.conjunctions = [
            "而且", "不仅如此", "另外", "同时", "此外",
            "因此", "所以", "总的来说", "总之", "简而言之",
            "换句话说", "具体来说", "值得注意的是", "特别是", "尤其是",
            "从另一个角度看", "更重要的是", "需要强调的是", "综上所述", "由此可见"
        ]

        # 添加结束语模板
        self.endings = [
            "这是一个很值得深入探讨的话题。",
            "这个问题需要我们持续关注和思考。",
            "希望这个回答对你有帮助。",
            "我们可以继续交流这个话题。",
            "这只是个人的见解，欢迎讨论。",
            "这个领域还有很多值得探索的地方。",
            "期待看到这方面更多的发展。",
            "让我们一起关注这个话题的发展。"
        ]

    def generate_question(self, topic):
        """生成问题"""
        template = random.choice(self.question_templates)
        return template.format(topic=topic)

    def generate_answer(self, topic):
        """生成回答"""
        # 选择主要内容
        main_content = random.choice(self.topics[topic])
        
        # 添加情感色彩
        emotion_type = random.choice(list(self.emotions.keys()))
        emotion_phrase = random.choice(self.emotions[emotion_type])
        
        # 添加连接词
        conjunction = random.choice(self.conjunctions)
        
        # 添加结束语
        ending = random.choice(self.endings)
        
        # 组合回答
        answer = f"{emotion_phrase}，{main_content}{conjunction}，{ending}"
        return answer

    def generate_dataset(self, num_samples, output_file):
        """生成数据集"""
        dataset = []
        
        print(f"正在生成{num_samples}个问答对...")
        for _ in tqdm(range(num_samples)):
            # 随机选择主题
            topic = random.choice(list(self.topics.keys()))
            
            # 生成问答对
            question = self.generate_question(topic)
            answer = self.generate_answer(topic)
            
            # 添加到数据集
            dataset.append({
                "question": question,
                "answer": answer,
                "topic": topic
            })
        
        # 保存到文件
        print(f"正在保存数据集到{output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"数据集生成完成！共{num_samples}个问答对已保存到{output_file}")

def main():
    # 创建数据集生成器
    generator = DatasetGenerator()
    
    # 生成训练集和验证集
    generator.generate_dataset(100000, 'train_dataset.json')
    generator.generate_dataset(10000, 'valid_dataset.json')

if __name__ == "__main__":
    main() 