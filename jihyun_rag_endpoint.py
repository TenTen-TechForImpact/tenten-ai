__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import json
import re
import boto3  # Import Boto3 for S3 access

import logging
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import requests


logging.basicConfig(level=logging.INFO)

import os

openai_api_key = os.getenv("OPENAI_API_KEY_1010")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

print(f"OpenAI API: {openai_api_key[:4]}********")
print(f"LangChain API Key: {langchain_api_key[:4]}********")

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["LANGCHAIN_PROJECT"] = "tech4impact_1010"
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# 시간 변환 함수
def ms_to_minutes_seconds_str(milliseconds):
    seconds = milliseconds // 1000
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{minutes}분 {remaining_seconds}초"

# 텍스트에서 시간 추출
def time_str_to_ms(time_str):
    match = re.match(r'(\d+)분 (\d+)초', time_str)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        return (minutes * 60 + seconds) * 1000
    else:
        logging.warning(f"시간 형식이 일치하지 않습니다: {time_str}")
        return 0

def lambda_handler(event, context):
    try:
        
        logging.info(f"Received event: {event}")
        print(event)
        # Extract the body from the event and parse it
        if 'body' not in event:
            raise ValueError("Request body is missing from the event")

        body = json.loads(event['body'])  # Parse the JSON body
        s3_url = body.get("s3_url")

        if not s3_url:
            raise ValueError("s3_url is missing from the event")

        # Parse the bucket name and key from the s3_url
        if not s3_url.startswith("s3://"):
            raise ValueError("Invalid S3 URL. It should start with 's3://'.")
        # Parse the bucket name and key from the s3_url
        if not s3_url.startswith("s3://"):
            raise ValueError("Invalid S3 URL. It should start with 's3://'.")

        s3_parts = s3_url.replace("s3://", "").split("/", 1)
        bucket_name = s3_parts[0]
        object_key = s3_parts[1]

        # Use Boto3 to fetch the JSON file from S3
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        data = json.loads(response['Body'].read().decode('utf-8'))
        
    
        utterances = data.get("utterances", [])
        subtopics = data.get("subtopics", [])
        
        # 문서 객체 생성
        documents = [Document(page_content=utterance['msg'], metadata={
            'start_at': utterance['start_at'],
            'duration': utterance['duration'],
            'speaker': utterance['spk'],
            'speaker_type': utterance['spk_type']
        }) for utterance in utterances]

        # 정렬
        documents = sorted(documents, key=lambda doc: doc.metadata['start_at'])
        full_text = ''
        for doc in documents:
                start_time_str = ms_to_minutes_seconds_str(doc.metadata['start_at'])
                full_text += f"[{start_time_str}] {doc.page_content}\n"
        
        
        # 모델 초기화
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=None)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        prompt = """
        당신은 전문적인 상담 세션을 분석하고 요약하는 AI입니다. 아래 상담 세션의 전체 스크립트를 주제별로 시간 순서에 따라 요약해 주세요. 다음 지침을 반드시 따르세요:
        
        1. 각 시간 구간은 **하나의 주요 질문** 또는 **하나의 주요 조언**으로 구성합니다. 하나의 구간에 여러 주제가 섞이지 않도록 주의해 주세요.
        2. 각 주제에 대해 시작 시간과 종료 시간을 명시합니다. 시작 시간은 해당 주제가 스크립트에서 처음 등장한 시점, 종료 시간은 마지막으로 등장한 시점을 기준으로 합니다.
        3. 각 요약은 다음 형식 중 **하나**로 작성합니다:
           - 질문인 경우: **Q.**으로 시작하여 질문 내용을 요약합니다.
           - 행동 변경이 필요한 경우: **[바꾸세요]**로 시작하여 변경이 필요한 행동을 요약합니다.
           - 권장 사항인 경우: **[이렇게 하세요]**로 시작하여 권장 사항을 요약합니다.
        
        **답변 예시:**
        **0분 42초 ~ 3분 15초**
        - Q. 오메가3 가 정말 유익한 거 맞나요?
        
        **5분 0초 ~ 7분 30초**
        - [바꾸세요] 당뇨 약은 중단하지 마세요.
        
        **10분 15초 ~ 12분 45초**
        - [이렇게 하세요] 약 복용을 지속하세요.
        
        **스크립트:**
        {text}
        
        **요약:**
        """

        human_prompt = PromptTemplate(
            input_variables=[],
            template="요약:"
        )
        rag_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",  # 또는 "map_reduce" 등 필요에 따라 변경 가능
                    retriever=retriever,
                    return_source_documents=True
                )

        # 전체 스크립트 요약 
        response = rag_chain({"query": prompt.format(text=full_text)})
        session_summary = response["result"]
        source_documents = response["source_documents"]

        subtopics = re.findall(r'\*\*(\d+분 \d+초) ~ (\d+분 \d+초)\*\*\n- (.+?)(?=\n\*\*\d+분|\Z)', session_summary, re.DOTALL)


        # 주제별 데이터 생성
        json_results = []
        for idx, (start_time_str, end_time_str, content) in enumerate(subtopics, 1):
            subtopic = content.strip()
            start_time_ms = time_str_to_ms(start_time_str)
            end_time_ms = time_str_to_ms(end_time_str)
            
            retrieved_docs = retriever.get_relevant_documents(subtopic)
            relevant_docs = [doc for doc in retrieved_docs if start_time_ms <= doc.metadata['start_at'] <= end_time_ms]
            
            # JSON 데이터 생성
            related_scripts = [{"time": ms_to_minutes_seconds_str(doc.metadata['start_at']), "content": doc.page_content} for doc in relevant_docs]
            json_result = {"topic_id": idx, "start_time": start_time_str, "end_time": end_time_str, "content": subtopic, "related_scripts": related_scripts}
            json_results.append(json_result)

        # 결과 반환
        return {
            "statusCode": 200,
            "body": json.dumps(json_results, ensure_ascii=False)
        }

    except Exception as e:
        logging.error(f"오류 발생: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
