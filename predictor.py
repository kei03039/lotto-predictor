import tensorflow as tf
import numpy as np
import os
from data_processor import get_all_lotto_history

class Predictor:
    def __init__(self, model_path='models/trained_model.h5'):
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            # 모델의 출력 형태 확인 (45개 노드인지 6개 노드인지)
            self.output_dim = self.model.output_shape[-1]
            print(f"모델 로드 완료 (출력 차원: {self.output_dim})")
        else:
            raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")
        self.history_df = get_all_lotto_history()

    def get_numbers(self):
        try:
            # 1. 노이즈 생성 (다양성 확보)
            noise_level = np.random.uniform(1.5, 4.0)
            noise = np.random.normal(0, noise_level, (1, 100))
            
            # 모델 예측
            raw_preds = self.model.predict(noise, verbose=0)[0]

            # 2. 모델 출력 타입에 따른 분기 처리 (오류 방지 핵심)
            if self.output_dim == 45:
                # [확률 분포형] 45개 노드 출력 시
                probabilities = raw_preds
            else:
                # [기존 숫자형] 6개 노드 출력 시 -> 45개 분포로 강제 변환
                probabilities = np.zeros(45)
                for p in raw_preds:
                    idx = int(np.clip(p * 44, 0, 44))
                    probabilities[idx] += 1.0
            
            # 3. 온도 조절 및 평탄화 (1과 45 고정 방지 및 다양성)
            temp = np.random.uniform(1.2, 2.5)
            probabilities = np.power(probabilities + 1e-10, 1.0 / temp)
            probabilities += np.random.uniform(0, probabilities.max() * 0.2, 45)
            probabilities /= probabilities.sum()

            # 4. 최근 당첨 번호 페널티 (최근 15회차)
            recent_15 = self.history_df.tail(15)[['no1','no2','no3','no4','no5','no6']].values.flatten()
            for n in recent_15:
                idx = int(n) - 1
                if 0 <= idx < 45:
                    probabilities[idx] *= 0.7

            # 5. 최종 6개 번호 무작위 추출
            selected_indices = np.random.choice(range(45), size=6, replace=False, p=probabilities)
            final_numbers = [int(idx + 1) for idx in selected_indices]

            return sorted(final_numbers)

        except Exception as e:
            print(f"번호 생성 중 오류 발생: {e}")
            # 오류 발생 시 완전 랜덤 번호라도 반환하여 시스템 중단 방지
            return sorted(np.random.choice(range(1, 46), 6, replace=False).tolist())
