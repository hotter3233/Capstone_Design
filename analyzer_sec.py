import cv2
import mediapipe as mp
import numpy as np
import copy 
from dataclasses import dataclass
from collections import Counter, deque 
from utils import calculate_angle_3d, calculate_angle_projected, extract_3d_pts, check_repetition, check_static, calculate_wrist_twist
from rula_engine import RULAEngine

@dataclass
class EvalResult:
    rula_score: int
    trunk: float
    lower_arm: float  
    neck: float
    wrist: float
    upper_arm: float
    twist: int
    flags: dict
    side: str = "Unknown"

def check_visibility(pts, threshold=0.5):
    """주요 관절 가시성 확인 (의도적 설계: index 제외)"""
    for key in ['shoulder', 'elbow', 'wrist', 'hip']:
        if pts[key][3] < threshold: return False
    return True


def calculate_hybrid_neck_angle(pts_n_l, pts_n_r):
    
    #양쪽 관절이 완벽히 보일 땐 '3D 중심축 투영'을, 가려짐이 발생하면 '단일 측면 YZ 투영'을 사용
    
    STRICT_THRESH = 0.85 # 가시성 기준
    
    # 귀(Ear)의 가시성도 중요하므로 조건에 추가
    left_vis = pts_n_l['shoulder'][3] > STRICT_THRESH and pts_n_l['hip'][3] > STRICT_THRESH and pts_n_l['ear'][3] > STRICT_THRESH
    right_vis = pts_n_r['shoulder'][3] > STRICT_THRESH and pts_n_r['hip'][3] > STRICT_THRESH and pts_n_r['ear'][3] > STRICT_THRESH
    
    #  MODE 1: 정면/대각선 (양쪽이 다 잘 보임) -> 3D 중심축 로직 가동
    if left_vis and right_vis:
        # 1. 뼈대 중심점 계산 
        ear_l, ear_r = np.array(pts_n_l['ear'][:3]), np.array(pts_n_r['ear'][:3])
        sh_l, sh_r = np.array(pts_n_l['shoulder'][:3]), np.array(pts_n_r['shoulder'][:3])
        hip_l, hip_r = np.array(pts_n_l['hip'][:3]), np.array(pts_n_r['hip'][:3])
        
        head_anchor = (ear_l + ear_r) / 2.0
        shoulder_center = (sh_l + sh_r) / 2.0
        torso_center = (hip_l + hip_r) / 2.0
        
        # 2. 3D 벡터 정의
        shoulder_line = sh_r - sh_l
        shoulder_line = shoulder_line / (np.linalg.norm(shoulder_line) + 1e-6)
        
        torso_axis = shoulder_center - torso_center
        neck_axis = head_anchor - shoulder_center
        
        # 3. 어깨선을 기준축으로 직교 투영 
        def project_perpendicular(v, normal):
            return v - normal * np.dot(v, normal)
            
        torso_in_plane = project_perpendicular(torso_axis, shoulder_line)
        neck_in_plane = project_perpendicular(neck_axis, shoulder_line)
        
        # 4. 투영된 척추와 목 벡터 간의 각도 도출
        t_norm = np.linalg.norm(torso_in_plane) + 1e-6
        n_norm = np.linalg.norm(neck_in_plane) + 1e-6
        
        cosine_angle = np.dot(torso_in_plane / t_norm, neck_in_plane / n_norm)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    #  MODE 2: 측면 (한쪽이 가려짐) ->  YZ 투영망
    active_pts = pts_n_l if left_vis else pts_n_r if right_vis else None
    
    if not active_pts: # 둘 다 STRICT를 못 넘었지만 기본(0.5)은 넘는 쪽
        if pts_n_l['shoulder'][3] > 0.5: active_pts = pts_n_l
        elif pts_n_r['shoulder'][3] > 0.5: active_pts = pts_n_r

    if active_pts:
        ear = active_pts['ear']
        shoulder = active_pts['shoulder']
        shoulder_up = [shoulder[0], shoulder[1] - 0.5, shoulder[2]]
        return calculate_angle_projected(ear, shoulder, shoulder_up, plane='YZ')
        
    return 0.0


def eval_side(engine, p_world, hand_landmarks, load_kg, pts_n_l=None, pts_n_r=None) -> EvalResult:
    shoulder, elbow, wrist = p_world['shoulder'], p_world['elbow'], p_world['wrist']
    hip, ear, index = p_world['hip'], p_world['ear'], p_world['index']

    shoulder_down = [shoulder[0], shoulder[1] + 0.5, shoulder[2]] 
    hip_up = [hip[0], hip[1] - 0.5, hip[2]]                       

    ua = calculate_angle_3d(elbow, shoulder, shoulder_down)
    tr = calculate_angle_projected(shoulder, hip, hip_up, plane='YZ')
    
    #  하이브리드 목 각도 호출
    if pts_n_l and pts_n_r:
        nk = calculate_hybrid_neck_angle(pts_n_l, pts_n_r)
    else:
        shoulder_up = [shoulder[0], shoulder[1] - 0.5, shoulder[2]]
        nk = calculate_angle_projected(ear, shoulder, shoulder_up, plane='YZ')
    
    la_raw = calculate_angle_3d(shoulder, elbow, wrist)
    la = 180 - la_raw if la_raw else 0
    
    wr_raw = calculate_angle_3d(elbow, wrist, index)
    wr_internal = 180 - wr_raw if wr_raw else 0
    
    if load_kg > 0 and wr_internal > 30:
        wr = 30 + (wr_internal - 30) * 0.2
    else:
        wr = wr_internal

    w_t = calculate_wrist_twist(hand_landmarks)
    
    is_neck_twisted = abs(ear[2] - shoulder[2]) > 0.20
    is_trunk_twisted = abs(shoulder[2] - hip[2]) > 0.25
    is_wrist_deviated = abs(index[0] - wrist[0]) > 0.15
    
    is_arm_abducted = abs(elbow[0] - shoulder[0]) > 0.20
    ear_shoulder_dist = np.linalg.norm(np.array(ear[:3]) - np.array(shoulder[:3]))
    is_shoulder_raised = ear_shoulder_dist < 0.15 
    is_upper_arm_penalty = is_arm_abducted or is_shoulder_raised
    
    u_s = engine.get_upper_arm_score(ua, is_abducted_or_raised=is_upper_arm_penalty)
    l_s = engine.get_lower_arm_score(la) 
    w_s = engine.get_wrist_score(wr, is_deviated=is_wrist_deviated)
    n_s = engine.get_neck_score(nk, is_twisted_or_bent=is_neck_twisted)
    t_s = engine.get_trunk_score(tr, is_twisted_or_bent=is_trunk_twisted)
    
    if load_kg < 2: rt_f_load = 0
    elif load_kg <= 10: rt_f_load = 1 
    else: rt_f_load = 3

    rula_score, _ = engine.calculate_final_score(
        u_s, l_s, w_s, w_t, n_s, t_s, 1, 
        m_a=0, f_a=rt_f_load, m_b=0, f_b=rt_f_load
    )
    
    flags = {
        "arm_abd": is_upper_arm_penalty, 
        "wr_dev": is_wrist_deviated, 
        "nk_tw": is_neck_twisted, 
        "tr_tw": is_trunk_twisted
    }
    return EvalResult(rula_score, tr, la, nk, wr, ua, w_t, flags)

def generate_final_report(res, engine, leg_score, load_kg, worst_img, worst_sec, max_s):
    if not res["sec"]: 
        return {"summary": {"score": 1, "action": "데이터 없음", "total": 0}, "ts": res, "worst": {"img": None, "sec": 0, "score": 0}}

    if len(set(res["rula"])) == 1:
        top_indices = list(range(len(res["rula"])))
    else:
        p75_rula = np.percentile(res["rula"], 75)
        top_indices = [i for i, score in enumerate(res["rula"]) if score >= p75_rula]
    
    rep_ua = np.median([res["upper_arm"][i] for i in top_indices])
    rep_la = np.median([res["elbow"][i] for i in top_indices]) 
    rep_wr = np.median([res["wrist"][i] for i in top_indices])
    rep_nk = np.median([res["neck"][i] for i in top_indices])
    rep_tr = np.median([res["trunk"][i] for i in top_indices])
    rep_twist = int(np.median([res["twist"][i] for i in top_indices]))
    
    rep_side = Counter([res["side"][i] for i in top_indices]).most_common(1)[0][0]
    
    half_len = len(top_indices) / 2
    rep_flags = {
        "arm_abd": sum(res["flags"][i]["arm_abd"] for i in top_indices) > half_len,
        "wr_dev": sum(res["flags"][i]["wr_dev"] for i in top_indices) > half_len,
        "nk_tw": sum(res["flags"][i]["nk_tw"] for i in top_indices) > half_len,
        "tr_tw": sum(res["flags"][i]["tr_tw"] for i in top_indices) > half_len,
    }
    
    m_use = 1 if (check_repetition(res["elbow"]) or check_static(res["trunk"])) else 0
    
    if load_kg < 2: final_f_load = 0
    elif 2 <= load_kg <= 10: final_f_load = 2 if m_use == 1 else 1 
    else: final_f_load = 3
    
    fin_r, fin_a = engine.calculate_final_score(
        engine.get_upper_arm_score(rep_ua, is_abducted_or_raised=rep_flags["arm_abd"]), 
        engine.get_lower_arm_score(rep_la), 
        engine.get_wrist_score(rep_wr, is_deviated=rep_flags["wr_dev"]), 
        rep_twist, 
        engine.get_neck_score(rep_nk, is_twisted_or_bent=rep_flags["nk_tw"]),
        engine.get_trunk_score(rep_tr, is_twisted_or_bent=rep_flags["tr_tw"]), 
        leg_score, m_use, final_f_load, m_use, final_f_load
    )
    
    risk_details = {
        "worst_side": f"{'왼쪽' if rep_side == 'Left' else '오른쪽' if rep_side == 'Right' else '알 수 없음'} 팔 집중 분석",
        "wrist_twist": "발견 (+1 감점)" if rep_twist == 2 else "정상",
        "wrist_deviation": "발견 (+1 감점)" if rep_flags["wr_dev"] else "정상",
        "neck_twist": "발견 (+1 감점)" if rep_flags["nk_tw"] else "정상",
        "trunk_twist": "발견 (+1 감점)" if rep_flags["tr_tw"] else "정상",
        "arm_abduction": "발견 (+1 감점)" if rep_flags["arm_abd"] else "정상",
        "repetition_or_static": "발견 (+1 감점)" if m_use == 1 else "정상",
        "heavy_load": f"해당 (+{final_f_load} 감점)" if final_f_load > 0 else "정상"
    }
    
    return {
        "summary": {"score": fin_r, "action": fin_a, "total": len(res["sec"]), "risk_details": risk_details}, 
        "ts": res, 
        "worst": {"img": worst_img, "sec": worst_sec, "score": max_s}
    }

def analyze_video_per_second(video_path, load_kg=0, leg_score=1):
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(int(fps), 1) if fps > 0 else 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    res = {"sec": [], "trunk": [], "elbow": [], "upper_arm": [], "neck": [], "wrist": [], "twist": [], "rula": [], "flags": [], "side": []}
    max_s, worst_img, worst_sec = -1, None, 0
    engine = RULAEngine()
    
    #  [추가] 과거 데이터를 기억할 버퍼 생성 (최대 3개 유지)
    history = deque(maxlen=3)
    
    last_valid_pts = None 
    has_data, missing_count = False, 0
    f_idx = 0
    
    try:
        while f_idx < total_frames:
            for _ in range(interval - 1):
                cap.grab()
                f_idx += 1
                
            ret, frame = cap.read()
            f_idx += 1
            if not ret: break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out = holistic.process(rgb)
            curr_sec = f_idx // interval
            
            is_hold_frame = False 
            eval_res = None
            draw_pts = None
            
            if out.pose_world_landmarks and out.pose_landmarks:
                w_lms = out.pose_world_landmarks.landmark 
                n_lms = out.pose_landmarks.landmark       
                
                pts_w_l = extract_3d_pts(w_lms, "left")
                pts_n_l = extract_3d_pts(n_lms, "left")
                pts_w_r = extract_3d_pts(w_lms, "right")
                pts_n_r = extract_3d_pts(n_lms, "right")
                
                vis_l, vis_r = check_visibility(pts_n_l), check_visibility(pts_n_r)
                
                #  상체가 가려졌을 때: 궤적 예측 로직 발동
                if not vis_l and not vis_r:
                    missing_count += 1
                    
                    if has_data and missing_count <= 3 and len(history) > 0:
                        # 1. 깊은 복사(Deep Copy)를 통해 원본 메모리 오염 방지
                        eval_res = copy.deepcopy(history[-1])
                        
                        # 2. 버퍼에 데이터가 2개 이상 있다면 속도(Velocity)를 계산하여 각도 예측
                        if len(history) >= 2:
                            prev1 = history[-1] # 직전 데이터
                            prev2 = history[-2] # 그 전 데이터
                            
                            # 예측된 미래 각도 = 현재 각도 + (가속도 * 가려진 누적 횟수)
                            eval_res.trunk += (prev1.trunk - prev2.trunk) * missing_count
                            eval_res.lower_arm += (prev1.lower_arm - prev2.lower_arm) * missing_count
                            eval_res.neck += (prev1.neck - prev2.neck) * missing_count
                            eval_res.wrist += (prev1.wrist - prev2.wrist) * missing_count
                            eval_res.upper_arm += (prev1.upper_arm - prev2.upper_arm) * missing_count
                            
                            # (안전망) 물리적으로 불가능한 각도 클리핑 방어
                            eval_res.lower_arm = max(0.0, min(180.0, eval_res.lower_arm))
                            eval_res.wrist = max(0.0, min(180.0, eval_res.wrist))
                            eval_res.upper_arm = max(0.0, min(180.0, eval_res.upper_arm))
                            
                        draw_pts = last_valid_pts
                        is_hold_frame = True 
                    else:
                        has_data = False
                        continue 
                        
                #  정상적으로 가시성이 확보되었을 때
                else:
                    missing_count = 0
                    
                    eval_l = eval_side(engine, pts_w_l, out.left_hand_landmarks, load_kg, pts_n_l, pts_n_r) if vis_l else None
                    if eval_l: eval_l.side = "Left"
                    
                    eval_r = eval_side(engine, pts_w_r, out.right_hand_landmarks, load_kg, pts_n_l, pts_n_r) if vis_r else None
                    if eval_r: eval_r.side = "Right"
                    
                    if eval_l and eval_r:
                        if eval_l.rula_score >= eval_r.rula_score: eval_res, draw_pts = eval_l, pts_n_l
                        else: eval_res, draw_pts = eval_r, pts_n_r
                    elif eval_l: eval_res, draw_pts = eval_l, pts_n_l
                    else: eval_res, draw_pts = eval_r, pts_n_r
                    
                    #  [추가] 정상 평가가 끝나면 결과를 메모리에 밀어 넣음
                    history.append(eval_res)
                    last_valid_pts = draw_pts
                    has_data = True
                
                if eval_res is not None:
                    res["sec"].append(curr_sec)
                    res["trunk"].append(eval_res.trunk)
                    res["elbow"].append(eval_res.lower_arm) 
                    res["upper_arm"].append(eval_res.upper_arm)
                    res["neck"].append(eval_res.neck)
                    res["wrist"].append(eval_res.wrist)
                    res["twist"].append(eval_res.twist)
                    res["rula"].append(eval_res.rula_score)
                    res["flags"].append(eval_res.flags)
                    res["side"].append(eval_res.side)
                    
                    if not is_hold_frame and eval_res.rula_score > max_s:
                        max_s, worst_sec = eval_res.rula_score, curr_sec
                        tmp = frame.copy()
                        h, w, _ = frame.shape
                        for p1, p2 in [('shoulder', 'elbow'), ('elbow', 'wrist'), ('wrist', 'index'), ('ear', 'shoulder'), ('shoulder', 'hip')]:
                            cv2.line(tmp, (int(draw_pts[p1][0]*w), int(draw_pts[p1][1]*h)), (int(draw_pts[p2][0]*w), int(draw_pts[p2][1]*h)), (0, 255, 0), 3)
                            cv2.circle(tmp, (int(draw_pts[p1][0]*w), int(draw_pts[p1][1]*h)), 5, (255, 255, 255), -1)
                        cv2.putText(tmp, f"Max RULA: {max_s}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        worst_img = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
        holistic.close()
    
    return generate_final_report(res, engine, leg_score, load_kg, worst_img, worst_sec, max_s)