from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 데이터셋 로드하기
# [[your code]
data = load_boston()
X = data.data
y = data.target

# 훈련용 데이터셋 나누기
# [[your code]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state=30)

# 훈련하기
# [[your code]
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 예측하기
# [[your code]
y_pred = model.predict(X_test)

# 정답률 출력하기
# [[your code]
print("accuracy = ", accuracy_score(y_test, y_pred))
