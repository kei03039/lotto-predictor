from flask import Flask, render_template, jsonify, request
import requests
import pymysql
import os

app = Flask(__name__)

# 🌐 대현님이 발굴하신 최신 공식 API 주소 기반의 점진적 자동화 엔진
def sync_lotto_database():
    from data_processor import get_db_connection
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. 현재 DB에 저장된 최고 회차 확인 (처음엔 비어있으므로 1회차부터 시작)
    cursor.execute("SELECT MAX(draw_no) as max_no FROM lotto_history")
    result = cursor.fetchone()
    start_no = (result['max_no'] + 1) if result['max_no'] else 1
    
    current_no = start_no
    print(f"🔄 [MariaDB] 대현님 전용 API 가동: {current_no}회차부터 최신 회차까지 실시간 추적...")
    
    while True:
        url = f"https://www.dhlottery.co.kr/lt645/selectPstLt645Info.do?srchLtEpsd={current_no}"
        try:
            res_data = requests.get(url, timeout=5).json()
        except Exception as e:
            print(f"❌ API 통신 지연 또는 완료: {e}")
            break
            
        # 데이터가 없거나 리스트가 비어있으면 (이번 주 회차를 넘어서면) 반복 종료
        if not res_data.get("data") or not res_data["data"].get("list"):
            print(f"✅ [MariaDB] 최신 회차까지 데이터 적재 완료! 현재 최종 회차: {current_no - 1}회")
            break
            
        # 대현님이 찍어주신 실제 JSON 데이터 구조 그대로 매핑 ('list'의 0번째 방 추출)
        lotto_info = res_data["data"]["list"][0]
        
        draw_no = lotto_info["ltEpsd"]       # 회차 (1000)
        raw_date = str(lotto_info["ltRflYmd"]) # 날짜 (20220129)
        # YYYYMMDD를 YYYY-MM-DD 포맷으로 변환하여 DB에 이쁘게 저장
        draw_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}" 
        
        # 번호 1~6번 및 보너스 번호 매핑
        num1 = lotto_info["tm1WnNo"]
        num2 = lotto_info["tm2WnNo"]
        num3 = lotto_info["tm3WnNo"]
        num4 = lotto_info["tm4WnNo"]
        num5 = lotto_info["tm5WnNo"]
        num6 = lotto_info["tm6WnNo"]
        bonus = lotto_info["bnsWnNo"]
        
        # MariaDB에 적재 수행
        sql = """
            INSERT INTO lotto_history (draw_no, draw_date, num1, num2, num3, num4, num5, num6, bonus)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (draw_no, draw_date, num1, num2, num3, num4, num5, num6, bonus))
        
        # 50회차마다 중간 커밋 및 로그 출력
        if current_no % 50 == 0:
            conn.commit()
            print(f"📦 로또 역사 적재 중... ({current_no}회차 완료)")
            
        current_no += 1

    conn.commit()
    cursor.close()
    conn.close()

# 🔥 앱 기동 시 자동 동기화 가동
with app.app_context():
    try:
        sync_lotto_database()
    except Exception as e:
        print(f"⚠️ 동기화 엔진 구동 중 예외 발생: {e}")

# 동기화 완료 후 AI 예측기 안전하게 가동
from predictor import Predictor
predictor = Predictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate')
def generate():
    all_results = []
    try:
        history_df = get_all_lotto_history()

        for _ in range(10):
            nums = predictor.get_numbers()
            user_nums_set = set(nums)

            stat = {
                "match3": 0, "match4": 0, "match5": 0, "match6": 0,
                "last_ro": 0, "last_matches": []
            }

            for _, row in history_df.iterrows():
                draw = {int(row['no1']), int(row['no2']), int(row['no3']),
                        int(row['no4']), int(row['no5']), int(row['no6'])}
                intersect = user_nums_set.intersection(draw)
                matches = len(intersect)
                ro_val = int(row['ro'])

                if matches == 3: stat["match3"] += 1
                elif matches == 4: stat["match4"] += 1
                elif matches == 5: stat["match5"] += 1
                elif matches == 6: stat["match6"] += 1

                if matches >= 4 and ro_val > stat["last_ro"]:
                    stat["last_ro"] = ro_val
                    stat["last_matches"] = sorted(list(intersect))

            all_results.append({"numbers": nums, "stats": stat})

        return jsonify({"results": all_results})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/check_history')
def check_history():
    nums_str = request.args.get('nums', "")
    if not nums_str: return "데이터 없음", 400
    try:
        user_nums = set(map(int, nums_str.split(',')))
        history_df = get_all_lotto_history()
        results = []
        for _, row in history_df.iterrows():
            draw_nums = {int(row['no1']), int(row['no2']), int(row['no3']),
                         int(row['no4']), int(row['no5']), int(row['no6'])}
            match_count = len(user_nums.intersection(draw_nums))
            if match_count >= 3:
                results.append({'ro': int(row['ro']), 'nums': sorted(list(draw_nums)), 'match': match_count})
        return render_template('result.html', user_nums=sorted(list(user_nums)),
                               results=sorted(results, key=lambda x: x['match'], reverse=True))
    except Exception as e:
        return f"오류: {e}", 500

from data_processor import get_all_lotto_history

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
