import numpy as np
from scipy.signal import find_peaks

def calculate_angle_3d(a, b, c):
    a = np.array(a[:3])
    b = np.array(b[:3])
    c = np.array(c[:3])
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def calculate_angle_projected(a, b, c, plane='YZ'):
    a = np.array(a[:3])
    b = np.array(b[:3])
    c = np.array(c[:3])
    
    if plane == 'YZ':
        a_proj = np.array([a[1], a[2]])
        b_proj = np.array([b[1], b[2]])
        c_proj = np.array([c[1], c[2]])
    elif plane == 'XY':
        a_proj = np.array([a[0], a[1]])
        b_proj = np.array([b[0], b[1]])
        c_proj = np.array([c[0], c[1]])
    elif plane == 'XZ':
        a_proj = np.array([a[0], a[2]])
        b_proj = np.array([b[0], b[2]])
        c_proj = np.array([c[0], c[2]])
    else:
        return calculate_angle_3d(a, b, c)

    ba = a_proj - b_proj
    bc = c_proj - b_proj
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def extract_3d_pts(landmarks, side="right"):
    mapping = {
        "right": {"shoulder": 12, "elbow": 14, "wrist": 16, "hip": 24, "ear": 8, "index": 20},
        "left": {"shoulder": 11, "elbow": 13, "wrist": 15, "hip": 23, "ear": 7, "index": 19}
    }
    
    pts = {}
    for key, idx in mapping[side].items():
        lm = landmarks[idx]
        pts[key] = [lm.x, lm.y, lm.z, lm.visibility]
    return pts

def check_repetition(angles, window_sec=60, fps=1, threshold=15):
    """
    반복 작업 여부 판별 (분당 4회 이상 유사한 패턴 반복 시 True)
    
    💡 계약(Contract): 
    이 함수의 'angles' 배열 매개변수는 반드시 초당 1프레임(1fps)으로 
    다운샘플링된 시계열 각도 데이터가 들어와야 정확한 환산이 이루어집니다.
    """
    if not angles: return False
    
    sec_length = len(angles) / fps
    if sec_length < 10:  
        return False
        
    peaks, _ = find_peaks(angles, prominence=threshold)
    estimated_peaks_per_min = len(peaks) * (60 / sec_length)
    
    return estimated_peaks_per_min >= 4

def check_static(angles, window_sec=60, fps=1, variance_thresh=5.0):
    if not angles: return False
    
    sec_length = len(angles) / fps
    if sec_length < 10:
        return False
        
    angles_np = np.array(angles)
    
    chunk_size = int(10 * fps) 
    if chunk_size == 0 or len(angles_np) < chunk_size: return False
    
    static_chunks = 0
    total_chunks = len(angles_np) // chunk_size
    
    for i in range(total_chunks):
        chunk = angles_np[i * chunk_size : (i+1) * chunk_size]
        if np.var(chunk) < variance_thresh:
            static_chunks += 1
            
    return (static_chunks / total_chunks) >= 0.5 if total_chunks > 0 else False

def calculate_wrist_twist(hand_landmarks):
    if not hand_landmarks:
        return 1 
        
    wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z])
    index_mcp = np.array([hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y, hand_landmarks.landmark[5].z])
    pinky_mcp = np.array([hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y, hand_landmarks.landmark[17].z])
    
    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist
    normal = np.cross(v1, v2)
    normal = normal / (np.linalg.norm(normal) + 1e-6)
    
    z_axis = np.array([0, 0, 1])
    angle = np.arccos(np.clip(np.dot(normal, z_axis), -1.0, 1.0))
    angle_deg = np.degrees(angle)
    
    if angle_deg < 45 or angle_deg > 135:
        return 2 
    else:
        return 1