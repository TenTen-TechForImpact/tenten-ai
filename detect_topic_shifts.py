import json
from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def detect_topic_shifts(utterances):
    chunks = []
    current_chunk = ''
    last_category = None

    for utterance in utterances:
        prompt = f"""
        아래 대화 내용이 이전 내용과 같은 주제인지, 아니면 다른 주제로 전환되었는지 분석하세요.
        대화: "{utterance['msg']}"

        아래의 카테고리 중, 이전 대화의 카테고리로부터 달라졌는지를 고려하여 분류하세요:
        1. 환자의 생활습관 진단, 복용하던 일반의약품
        2. 복약방법 (복용량, 횟수, 시간대)
        3. 약물 효과 (약물의 작용방식 설명)
        4. 부작용 및 경고
        5. 질병 관리 조언
        6. 기타 질의응답
        7. 필요 없는 정보 (가정사, 잡담)
        
        답변은 '같은 주제' 또는 '다른 주제'로 해주세요.
        """

        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=10)

        category = response.choices[0].message.content.strip()

        if category == '다른 주제':
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = f"[{utterance['speaker']}] {utterance['msg']}"
        else:
            current_chunk += f" [{utterance['speaker']}] {utterance['msg']}"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
