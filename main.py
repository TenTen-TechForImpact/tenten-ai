import json
from detect_topic_shifts import detect_topic_shifts
from categorize_chunks import categorize_chunks
from generate_summary import generate_summary

with open('simulation1.json', 'r') as file:
    utterances = json.load(file)['utterances']

for utterance in utterances:
    utterance['speaker'] = '약사' if utterance['spk'] == 0 else '환자'

# Step 1: 주제 전환 감지 및 문단 분할
chunks = detect_topic_shifts(utterances)

# Step 2: 각 문단 카테고리화
categorized_chunks = categorize_chunks(chunks)

# # Step 3: 카테고리별 요약 생성
summaries = generate_summary(categorized_chunks)

categories = {'1. 환자의 생활습관 진단, 복용하던 일반의약품':[], '2. 복약방법 (복용량, 횟수, 시간대)':[], '3. 약물 효과: 약물의 작용방식 설명':[], '4. 부작용 및 경고':[], '5. 질병 관리 조언':[], '6. 기타 질의응답':[], '7. 필요 없는 정보 (가정사, 잡담)':[]}

for summary in summaries:
    categories[summary['category']].append(summary['summary'])

for category, summaries in categories.items():
    print(f"Category {category}:")
    for summary in summaries:
        print(summary)
    print()
