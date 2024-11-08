from openai import OpenAI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# API 설정 및 클라이언트 초기화
API_KEY = 'API_KEY'
client = OpenAI(api_key=API_KEY)

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model).data[0].embedding
    return response

def extract_entity(query):
    return 

def find_best_match(query):
    entity = extract_entity(query)
    query_embedding = get_embedding(entity)
    similarities = cosine_similarity([query_embedding], df['ada_embedding'].tolist())
    return df['entity'].iloc[np.argmax(similarities)]

# 예제 데이터 (엔티티와 설명 분리)
data = {
    'entity': ["서울대학교", "팬더", "피아노", '인천시논현동', '강남구논현동'],
    'description': ["서울시 관악구에 위치한 학교입니다.", "주로 대나무를 먹으며 살고 있습니다.", "치는 것은 매우 즐거운 경험이 될 수 있습니다.", '인천구 논현동', '강남구 논현동']
}
df = pd.DataFrame(data)
df['ada_embedding'] = df['entity'].apply(get_embedding)  # 엔티티에 대한 임베딩만 생성

query = "서울대학교 근처 맛집 알려줘"
best_match_entity = find_best_match(query)
print("Best match entity:", best_match_entity)



from openai import OpenAI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# API 설정 및 클라이언트 초기화
API_KEY = 'API_KEY'
client = OpenAI(api_key=API_KEY)

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response['data'][0]['embedding']

def extract_entity(query):
    return "강남에 있는 논현동"

def find_best_match(query):
    entity = extract_entity(query)
    query_embedding = get_embedding(entity)
    
    entity_embeddings = df['contextual_embedding'].tolist()
    similarities = cosine_similarity([query_embedding], entity_embeddings)
    
    return df['entity'].iloc[np.argmax(similarities)]

# 예제 데이터 (엔티티와 설명 분리)
data = {
    'entity': ["서울대학교", "팬더", "피아노", '인천시 논현동', '강남구 논현동'],
    'description': ["서울시 관악구에 위치한 학교입니다.", "주로 대나무를 먹으며 살고 있습니다.", "치는 것은 매우 즐거운 경험이 될 수 있습니다.", '인천시에 있는 논현동', '강남구에 있는 논현동']
}
df = pd.DataFrame(data)

# 엔티티와 설명을 함께 입력하여 문맥 임베딩 생성
df['contextual_embedding'] = df.apply(lambda row: get_embedding(f"{row['description']} {row['entity']}"), axis=1)

query = "서울대학교 근처 맛집 알려줘"
best_match_entity = find_best_match(query)
print("Best match entity:", best_match_entity)
