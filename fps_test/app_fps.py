import streamlit as st
import tempfile, os
import pandas as pd
import json
from analyzer_fps import analyze_video_per_frame


st.set_page_config(page_title="BaroBon - RULA Dashboard", layout="wide")
st.title("바로본(BaroBon) ⏱️ RULA평가 기준 동작 분석 시스템")

with st.sidebar:
    st.markdown("### ⚙️ 분석 환경 설정")
    st.markdown("---")
    
    load_kg = st.number_input(
        "작업물 무게 (kg)", 
        min_value=0.0, 
        max_value=50.0, 
        value=5.0, 
        step=0.5,
        help="2kg 이상, 10kg 초과 등 무게에 따라 RULA 가중치가 다르게 적용됩니다."
    )
    
    leg_condition = st.selectbox(
        "다리 지지 상태",
        options=["안정적 지지 (양발 체중 분산)", "불안정 / 한쪽 발 지지"],
        index=0
    )
    leg_score = 1 if "안정적" in leg_condition else 2
    
    st.markdown("---")
    st.info("RULA 분석 안내\n이 시스템은 상체(상완, 전완, 손목, 목, 몸통)의 3D 각도를 분석하여 위험 등급을 산출합니다.")

up = st.file_uploader("분석할 영상을 업로드하세요", type=['mp4', 'mov', 'avi'])

if up:
    if st.button("AI 분석 시작", type="primary"):
        with st.spinner('초 단위 스캔 및 위험 순간 포착 중...'):
            
            t = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            t.write(up.read())
            path = t.name
            t.close() 
            
            data = analyze_video_per_frame(path, load_kg=load_kg, leg_score=leg_score)
            st.success("분석 완료!")

            
            video_length = data['ts']['sec'][-1] if data['ts']['sec'] else 0
            if video_length < 10:
                st.warning("⏱️ **데이터 부족 경고:** 영상 길이가 10초 미만입니다. 누적 데이터 부족으로 인해 '반복 작업' 및 '정적 자세' 판정(최소 10초 윈도우 요구)이 생략되었을 수 있습니다. 정확한 평가를 위해 15초 이상의 영상을 권장합니다.")

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
                        "arm_abduction": "💪 팔꿈치 들림 (외전)",
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
            st.write("다른 시스템(DB, LLM, 외부 API)으로 전송하기 위한 순수 분석 데이터입니다.")
            
            export_data = {
                "metadata": {
                    "worker_load_kg": load_kg,
                    "leg_condition_score": leg_score
                },
                "summary": data['summary'],
                "time_series_data": data['ts'],
                "peak_risk_event": {
                    "second": data['worst']['sec'],
                    "score": data['worst']['score']
                }
            }
            
            json_string = json.dumps(export_data, ensure_ascii=False, indent=4)
            
            col_json1, col_json2 = st.columns(2)
            
            with col_json1:
                st.download_button(
                    label="📥 분석 결과 JSON 파일 다운로드",
                    data=json_string,
                    file_name="barobon_analysis_result.json",
                    mime="application/json",
                    type="primary"
                )
                
            with col_json2:
                with st.expander("JSON 원본 데이터 보기"):
                    st.code(json_string, language='json')

            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass