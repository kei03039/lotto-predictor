import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import pandas as pd
import data_processor

# 데이터 전처리: 파일 검사를 생략하고 데이터 프로세서를 통해 MariaDB 실시간 연동
def load_and_preprocess_data():
    # 데이터 프로세서의 신규 DB 수집 함수 호출
    df = data_processor.get_all_lotto_history()
    
    # 최근 500회차 패턴만 추출
    df_recent = df.tail(500)
    raw_data = df_recent[['no1', 'no2', 'no3', 'no4', 'no5', 'no6']].values

    processed_data = []
    for row in raw_data:
        label = np.zeros(45)
        for num in row:
            if 1 <= int(num) <= 45:
                label[int(num)-1] = 1.0 / 6.0
        processed_data.append(label)

    return np.array(processed_data)

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
        layers.Dense(45, activation='softmax')
    ])
    return model

def build_discriminator():
    model = models.Sequential([
        layers.Dense(512, input_dim=45),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        layers.Dense(256),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])
    return model

def train():
    dataset = load_and_preprocess_data()

    generator = build_generator()
    discriminator = build_discriminator()
    discriminator.trainable = False

    gan_input = layers.Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = models.Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

    epochs = 4000  
    batch_size = 32
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    print("자릿수 고정 해결을 위한 확률 분포 학습을 시작합니다...")

    for epoch in range(epochs):
        idx = np.random.randint(0, dataset.shape[0], batch_size)
        real_dist = dataset[idx]
        d_loss_real = discriminator.train_on_batch(real_dist, real)

        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_dist = generator.predict(noise, verbose=0)
        d_loss_fake = discriminator.train_on_batch(gen_dist, fake)

        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan.train_on_batch(noise, real)

        if epoch % 500 == 0:
            print(f"Epoch {epoch}/{epochs} [D loss: {d_loss_real[0]:.4f}] [G loss: {g_loss:.4f}]")

    if not os.path.exists('models'):
        os.makedirs('models')
    generator.save('models/trained_model.h5')
    print("학습 완료! 'models/trained_model.h5'가 업데이트되었습니다.")

if __name__ == "__main__":
    train()
