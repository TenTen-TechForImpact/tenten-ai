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
