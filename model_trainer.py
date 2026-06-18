import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import pandas as pd

# 1. 데이터 전처리: 당첨 번호를 45차원의 확률 분포(Multi-hot encoding)로 변환
def load_and_preprocess_data(filepath='data/lotto.csv'):
    if not os.path.exists(filepath):
        raise FileNotFoundError("lotto.csv 파일이 필요합니다.")
    
    df = pd.read_csv(filepath)
    # 최근 500회차의 패턴이 현재 추첨 기계의 특성을 가장 잘 반영함
    df_recent = df.tail(500)
    raw_data = df_recent[['no1', 'no2', 'no3', 'no4', 'no5', 'no6']].values
    
    # 6개 숫자를 45개의 칸 중 해당 번호만 1로 표시하는 방식으로 변환
    processed_data = []
    for row in raw_data:
        # 1.0을 6개 숫자에 나눠주어 전체 합이 1인 확률 분포처럼 만듦
        label = np.zeros(45)
        for num in row:
            if 1 <= int(num) <= 45:
                label[int(num)-1] = 1.0 / 6.0 
        processed_data.append(label)
        
    return np.array(processed_data)

# 2. 생성자(Generator) 설계: 100개의 노이즈로부터 45개 번호의 '기운'을 생성
def build_generator():
    model = models.Sequential([
        layers.Dense(256, input_dim=100),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        
        layers.Dense(1024),
        layers.LeakyReLU(alpha=0.2),
        
        # 출력층: 45개 번호 각각의 확률 분포 생성 (Softmax로 전체 합 1 유지)
        layers.Dense(45, activation='softmax') 
    ])
    return model

# 3. 판별자(Discriminator) 설계: 생성된 분포가 실제 당첨 패턴과 유사한지 판별
def build_discriminator():
    model = models.Sequential([
        layers.Dense(512, input_dim=45),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        
        layers.Dense(256),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        
        layers.Dense(1, activation='sigmoid') # 진짜/가짜 판별
    ])
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])
    return model

def train():
    # 데이터 준비
    dataset = load_and_preprocess_data()
    
    # 모델 빌드 및 컴파일
    generator = build_generator()
    discriminator = build_discriminator()
    discriminator.trainable = False

    gan_input = layers.Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = models.Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

    # 학습 설정
    epochs = 4000  # 패턴 학습을 위해 에폭 상향
    batch_size = 32
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    print("자릿수 고정 해결을 위한 확률 분포 학습을 시작합니다...")

    for epoch in range(epochs):
        # 1. 실제 데이터 학습
        idx = np.random.randint(0, dataset.shape[0], batch_size)
        real_dist = dataset[idx]
        d_loss_real = discriminator.train_on_batch(real_dist, real)

        # 2. 가짜 데이터 생성 및 학습
        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_dist = generator.predict(noise, verbose=0)
        d_loss_fake = discriminator.train_on_batch(gen_dist, fake)

        # 3. 생성자 학습 (판별자를 속이도록)
        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan.train_on_batch(noise, real)

        if epoch % 500 == 0:
            print(f"Epoch {epoch}/{epochs} [D loss: {d_loss_real[0]:.4f}] [G loss: {g_loss:.4f}]")

    # 모델 저장
    if not os.path.exists('models'):
        os.makedirs('models')
    generator.save('models/trained_model.h5')
    print("학습 완료! 'models/trained_model.h5'가 업데이트되었습니다.")

if __name__ == "__main__":
    train()
