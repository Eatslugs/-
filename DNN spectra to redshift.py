import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
import matplotlib.pyplot as plt
import time

start = time.time()

# CSV 파일 로드
data = pd.read_csv("C:\\Users\\Administrator\\Desktop\\research\\galaxy data\\csv\\star_classification.csv")

# 입력 (features)와 출력 (target) 분리
X = data[['u','g','r','i','z']]
y = data['redshift']  # 적색편이

# 데이터 분할: 훈련 데이터와 테스트 데이터
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링 (정규화)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 정의
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(1)  # 출력층 (회귀 문제이므로 활성화 함수 없음)
])

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# 모델 요약 정보 출력
model.summary()

# 모델 훈련
history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2)

# 모델 평가
test_loss, test_mae = model.evaluate(X_test_scaled, y_test)
print(f'Test MAE: {test_mae}')

# 예측
predictions = model.predict(X_test_scaled)

# 예측 결과 출력
print(predictions)

print(round(time.time() - start,2))

plt.figure(figsize=(12, 6))

# 손실 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

# MAE 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('MAE over Epochs')
plt.legend()

plt.tight_layout()
plt.show()