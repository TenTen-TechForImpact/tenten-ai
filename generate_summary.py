# def generate_summary(categorized_chunks):
#     summary = {}

#     for item in categorized_chunks:
#         category = item['category']
#         if category not in summary:
#             summary[category] = []
#         summary[category].append(item['chunk'])

#     # 카테고리별로 요약을 연결해줍니다.
#     final_summary = {category: ' '.join(texts) for category, texts in summary.items()}

#     return final_summary

from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def generate_summary(categorized_chunks):
    summaries = []

    for item in categorized_chunks:
        prompt = f"""
        아래의 대화 내용 중, 환자에게 전달해야 할 중요한 내용을 요약하세요.:
        
        카테고리: {item['category']}
        대화 내용: "{item['chunk']}"

        요약본을 작성할 때, 중요한 약물 이름, 복약 방법, 주의 사항, 생활 습관에 대한 조언 등 주요 정보를 포함하세요. 
        불필요한 잡담이나 중복된 정보는 제외하고 간결하고 명확하게 요약해 주세요.

        요약:
        """

        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=150)

        summary = response.choices[0].message.content.strip()
        summaries.append({'category': item['category'], 'summary': summary})

    return summaries
