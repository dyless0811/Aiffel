'''
포켓몬 특성값으로 전설포켓몬 구분하는 모델
csv데이터: https://www.kaggle.com/abcsds/pokemon
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# 데이터 불러오기
csv_path = os.getenv("HOME") + "/desktop/project/practice/data/Pokemon.csv"
protocol_data = pd.read_csv(csv_path)

# 원본 데이터 유지
pokemon = protocol_data.copy()
print(pokemon.shape)

# 전설의 포켓몬 데이터셋
legendary = pokemon[pokemon["Legendary"] == True].reset_index(drop=True)

# 일반 포켓몬 데이터셋
normally = pokemon[pokemon["Legendary"] == False].reset_index(drop=True)

# types에 타입 종류 저장
types = list(set(pokemon["Type 1"]))

stats = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]

# 전설포켓몬들은 비슷한 이름을 가짐
# 문자열 컬럼 Name 처리
legendary["name_count"] = legendary["Name"].apply(lambda i: len(i))
normally["name_count"] = normally["Name"].apply(lambda i: len(i))
pokemon["name_count"] = pokemon["Name"].apply(lambda i: len(i))
pokemon["long_name"] = pokemon["name_count"] >= 10

pokemon["Name_nospace"] = pokemon["Name"].apply(lambda i: i.replace(" ", ""))
pokemon["name_isalpha"] = pokemon["Name_nospace"].apply(lambda i: i.isalpha())

pokemon = pokemon.replace(to_replace="Nidoran♀", value="Nidoran X")
pokemon = pokemon.replace(to_replace="Nidoran♂", value="Nidoran Y")
pokemon = pokemon.replace(to_replace="Farfetch'd", value="Farfetchd")
pokemon = pokemon.replace(to_replace="Mr. Mime", value="Mr Mime")
pokemon = pokemon.replace(to_replace="Porygon2", value="Porygon")
pokemon = pokemon.replace(to_replace="Ho-oh", value="Ho Oh")
pokemon = pokemon.replace(to_replace="Mime Jr.", value="Mime Jr")
pokemon = pokemon.replace(to_replace="Porygon-Z", value="Porygon Z")
pokemon = pokemon.replace(to_replace="Zygarde50% Forme", value="Zygarde Forme")

# 이름을 띄어쓰기와 대문자 기준으로 분리해서 토큰화
name = "CharizardMega Charizard X"
name_split = name.split(" ")
temp = name_split[0]  # Charizard와 Mega를 분리시키기 위해 따로 생성
tokens = re.findall('[A-Z][a-z]*', temp)  # findall로 대문자로 시작해서 소문자 한 개 이상의 패턴 찾기

tokens = []
for part_name in name_split:
    a = re.findall('[A-Z][a-z]*', part_name)
    tokens.extend(a)


# >>> ['Charizard', 'Mega', 'Charizard', 'X']

# 단어를 쪼개는 함수
def tokenize(name):
    name_split = name.split(" ")

    tokens = []
    for part_name in name_split:
        a = re.findall('[A-Z][a-z]*', part_name)
        tokens.extend(a)

    return np.array(tokens)


# 전설포켓몬의 Name에 함수를 적용
all_tokens = list(legendary["Name"].apply(tokenize).values)

token_set = []
for token in all_tokens:
    token_set.extend(token)

most_common = Counter(token_set).most_common(10)

for token, _ in most_common:
    pokemon[token] = pokemon["Name"].str.contains(token)

# 타입 1 & 2 범주형 데이터 전처리
for t in types:
    pokemon[t] = (pokemon["Type 1"] == t) | (pokemon["Type 2"] == t)

# 의미없는 컬럼과 문자열 컬럼 제외
features = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed',
            'Generation', 'name_count', 'long_name', 'Forme', 'Mega', 'Mewtwo',
            'Kyurem', 'Deoxys', 'Hoopa', 'Latias', 'Latios', 'Kyogre', 'Groudon',
            'Poison', 'Water', 'Steel', 'Grass', 'Bug', 'Normal', 'Fire', 'Fighting',
            'Electric', 'Psychic', 'Ghost', 'Ice', 'Rock', 'Dark', 'Flying', 'Ground',
            'Dragon', 'Fairy']

# 타켓(정답)
target = "Legendary"

X = pokemon[features]
y = pokemon[target]

# 데이터 셋 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

# 모델 적용과 예측
model = DecisionTreeClassifier(random_state=25)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 정답과 모델을 적용한 예측 값 비교
confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
