from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


def categorize_chunks(chunks):
    categorized_chunks = []

    for chunk in chunks:
        prompt = f"""
        아래 대화 내용을 읽고, 다음 카테고리 중 하나로 분류하세요:
        1. 환자의 생활습관 진단, 복용하던 일반의약품
        2. 복약방법 (복용량, 횟수, 시간대)
        3. 약물 효과: 약물의 작용방식 설명
        4. 부작용 및 경고
        5. 질병 관리 조언
        6. 기타 질의응답
        7. 필요 없는 정보 (가정사, 잡담)

        대화: "{chunk}"

        이 대화의 카테고리는?:
        """

        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=50)

        category = response.choices[0].message.content.strip()
        categorized_chunks.append({'chunk': chunk, 'category': category})

    return categorized_chunks
