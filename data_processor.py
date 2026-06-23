import pandas as pd
import numpy as np
import pymysql

# MariaDB 연결 공통 함수 (비밀번호 1234 반영)
def get_db_connection():
    return pymysql.connect(
        host='mariadb-service',  # 쿠버네티스 서비스 이름으로 내부 통신
        user='root',
        password='1234',         # 대현님이 변경하신 새 비밀번호!
        db='lottodb',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

def load_data():
    # AI 모델 학습용: 최신 500회차 확률 분포용 데이터 가져오기
    conn = get_db_connection()
    query = """
        SELECT num1 as no1, num2 as no2, num3 as no3, num4 as no4, num5 as no5, num6 as no6 
        FROM lotto_history 
        ORDER BY draw_no DESC 
        LIMIT 500
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    # AI 학습 패턴을 위해 최근 데이터가 아래로 가도록 정렬 뒤 넘파이 변환
    df = df.iloc[::-1].reset_index(drop=True)
    return df.values / 45.0

def get_all_lotto_history():
    # 웹 화면 및 통계용: 전체 역대 데이터 가져오기 (기존 코드의 컬럼명 매핑 유지)
    conn = get_db_connection()
    query = """
        SELECT draw_no as ro, num1 as no1, num2 as no2, num3 as no3, num4 as no4, num5 as no5, num6 as no6 
        FROM lotto_history 
        ORDER BY draw_no ASC
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df
