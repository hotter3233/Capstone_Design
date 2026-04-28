import streamlit as st
import tempfile, os
import pandas as pd
import json
import numpy as np  # 🔴 JSON 인코더용 Numpy 추가
from analyzer_sec import analyze_video_per_second

# 🔴 [JSON 에러 해결] Numpy 자료형을 일반 파이썬 자료형으로 변환해 주는 인코더 클래스
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):  # 에러의 주범인 bool_ 처리
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

st.set_page_config(page_title="BaroBon - RULA Dashboard", layout="wide")
st.title("바로본(BaroBon) ⏱️ 고도화 동작 분석 시스템")

# 세션 상태(Session State) 초기화
if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None
if "current_params" not in st.session_state:
    st.session_state.current_params = {}

with st.sidebar:
    st.markdown("### ⚙️ 분석 환경 설정")
    st.markdown("---")
    
    load_kg = st.number_input(
        "작업물 무게 (kg)", min_value=0.0, max_value=50.0, value=5.0, step=0.5
    )
    
    leg_condition = st.selectbox(
        "다리 지지 상태", options=["안정적 지지 (양발 체중 분산)", "불안정 / 한쪽 발 지지"], index=0
    )
    leg_score = 1 if "안정적" in leg_condition else 2
    
    st.markdown("---")
    st.info("RULA 분석 안내\n이 시스템은 상체(상완, 전완, 손목, 목, 몸통)의 3D 각도를 분석하여 위험 등급을 산출합니다.")

up = st.file_uploader("분석할 영상을 업로드하세요", type=['mp4', 'mov', 'avi'])

if up:
    if st.button("AI 분석 시작", type="primary"):
        with st.spinner('초 단위 스캔 및 위험 순간 포착 중...'):
            path = None
            try:
                # 임시 파일 생성
                t = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                t.write(up.read())
                path = t.name
                t.close() 
                
                # 엔진 구동
                data = analyze_video_per_second(path, load_kg=load_kg, leg_score=leg_score)
                
                # 세션 메모리에 저장
                st.session_state.analysis_data = data
                st.session_state.current_params = {"load_kg": load_kg, "leg_score": leg_score}
                st.success("분석 완료!")
                
            except Exception as e:
                st.error(f"분석 중 오류가 발생했습니다: {e}")
            finally:
                # 예외 발생 시에도 안전한 파일 정리
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass

# 분석 결과 렌더링
if st.session_state.analysis_data:
    data = st.session_state.analysis_data
    saved_load = st.session_state.current_params.get("load_kg", load_kg)
    saved_leg = st.session_state.current_params.get("leg_score", leg_score)

    # 파라미터 변경 감지 알림
    if saved_load != load_kg or saved_leg != leg_score:
        st.warning("⚠️ **설정이 변경되었습니다.** 변경된 가중치(무게/다리 지지)를 반영하려면 좌측의 [AI 분석 시작] 버튼을 다시 눌러주세요.")

    video_length = data['ts']['sec'][-1] if data['ts']['sec'] else 0
    if video_length < 10:
        st.warning("⏱️ **데이터 부족 경고:** 영상 길이가 10초 미만입니다. 최소 관측 시간 부족으로 '반복 작업' 및 '정적 자세' 패턴 분석이 제한될 수 있습니다.")

    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.subheader("📊 종합 평가")
        st.metric("최종 RULA 점수", f"{data['summary']['score']} 점")
        st.warning(f"권고: {data['summary']['action']}")
        
        if "risk_details" in data['summary']:
            st.markdown("#### 🔍 세부 위험 요인")
            key_mapping = {
                "worst_side": "💡 주요 위험 발생 위치",
                "wrist_twist": "🖐️ 손목 뒤틀림",
                "wrist_deviation": "🖐️ 손목 꺾임 (편위)",
                "neck_twist": "🧑 목 비틀림 / 측면 굽힘",
                "trunk_twist": "🩻 허리 비틀림 / 측면 굽힘",
                "arm_abduction": "💪 팔꿈치 들림 / 어깨 긴장",
                "repetition_or_static": "🔄 반복 / 정적 자세",
                "heavy_load": "📦 작업 하중"
            }
            for key, val in data['summary']["risk_details"].items():
                display_name = key_mapping.get(key, key)
                if "발견" in val or "해당" in val or "집중" in val:
                    st.markdown(f"- **{display_name}**: <span style='color:#ff4b4b; font-weight:bold;'>{val}</span> 🚨", unsafe_allow_html=True)
                else:
                    st.markdown(f"- **{display_name}**: <span style='color:#2e7b32;'>{val}</span> ✅", unsafe_allow_html=True)
    
    with c2:
        st.subheader("🚨 최대 위험 순간")
        if data['worst']['img'] is not None:
            # 🔴 [이미지 에러 해결] width 파라미터를 아예 빼버리고 최신 규격인 use_container_width=True 만 사용
            st.image(data['worst']['img'], caption=f"{data['worst']['sec']}초 시점 (점수: {data['worst']['score']})", use_container_width=True)

    st.divider()
    st.subheader("📈 시간대별 RULA 위험도 추이")
    df_r = pd.DataFrame({"시간": data['ts']['sec'], "RULA": data['ts']['rula']}).set_index("시간")
    st.line_chart(df_r, color="#FF4B4B")

    st.subheader("📉 시간대별 상세 관절 각도")
    df_a = pd.DataFrame({
        "시간": data['ts']['sec'], "허리": data['ts']['trunk'], 
        "팔꿈치": data['ts']['elbow'], "목": data['ts']['neck'], "손목": data['ts']['wrist']
    }).set_index("시간")
    st.line_chart(df_a)
    
    st.divider()
    st.subheader("🔗 데이터 내보내기 (JSON)")
    
    export_data = {
        "metadata": {
            "worker_load_kg": saved_load,
            "leg_condition_score": saved_leg
        },
        "summary": data['summary'],
        "time_series_data": data['ts'],
        "peak_risk_event": {
            "second": data['worst']['sec'],
            "score": data['worst']['score']
        }
    }
    
    # 🔴 [JSON 에러 해결] cls=NumpyEncoder를 적용하여 Numpy bool_ 및 float64 에러 원천 차단
    json_string = json.dumps(export_data, ensure_ascii=False, indent=4, cls=NumpyEncoder)
    
    col_json1, col_json2 = st.columns(2)
    with col_json1:
        st.download_button("📥 분석 결과 JSON 다운로드", data=json_string, file_name="barobon_analysis_result.json", mime="application/json", type="primary")
    with col_json2:
        with st.expander("JSON 원본 데이터 보기"):
            st.code(json_string, language='json')