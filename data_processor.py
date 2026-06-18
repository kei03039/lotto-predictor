import pandas as pd
import numpy as np

def load_data(filepath='data/lotto.csv'):
    df = pd.read_csv(filepath)
    # 최근 400회차 데이터만 사용하여 최신 트렌드 반영
    df_recent = df.tail(400)
    raw_data = df_recent[['no1', 'no2', 'no3', 'no4', 'no5', 'no6']]
    return raw_data.values / 45.0

def get_all_lotto_history(filepath='data/lotto.csv'):
    return pd.read_csv(filepath)
