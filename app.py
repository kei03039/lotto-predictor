from flask import Flask, render_template, jsonify, request
from predictor import Predictor
from data_processor import get_all_lotto_history
import os

app = Flask(__name__)
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
            
            # 등수별 카운트 초기화
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
                
                # 정확한 등수 판정 (4개=4등, 5개=3등)
                if matches == 3: stat["match3"] += 1
                elif matches == 4: stat["match4"] += 1
                elif matches == 5: stat["match5"] += 1
                elif matches == 6: stat["match6"] += 1
                
                # 하이라이트 표시용 (4등 이상 기록 중 최신 회차)
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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
