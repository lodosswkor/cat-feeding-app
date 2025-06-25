import torch

torch.classes.__path__ = []

import streamlit as st
import cv2
import os
import tempfile
from ultralytics import YOLO
import time
from collections import deque

model = YOLO("./yolov8n.pt")
st.title("YOLOv8 실시간 괭이 탐지 with 거리 추적")

# 고양이 거리 추적을 위한 전역 변수
if 'cat_distance_history' not in st.session_state:
    st.session_state.cat_distance_history = deque(maxlen=30)  # 최근 30프레임 저장
if 'last_cat_detection_time' not in st.session_state:
    st.session_state.last_cat_detection_time = 0



# 거리측정 함수 
def calculate_cat_distance(box, frame_width, frame_height):
    """고양이의 바운딩 박스 크기를 기반으로 거리 추정"""
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    box_width = x2 - x1
    box_height = y2 - y1
    box_area = box_width * box_height
    frame_area = frame_width * frame_height
    relative_size = box_area / frame_area
    
    # 상대적 크기를 거리 점수로 변환 (클수록 가까움)
    distance_score = relative_size * 1000
    return distance_score, box_area

# 움직임 분석 함수 
def analyze_cat_movement(distance_history):
    """고양이의 움직임 분석"""
    if len(distance_history) < 5:
        return "분석 중...", "neutral"
    
    recent_distances = list(distance_history)[-5:]  # 최근 5개
    if len(recent_distances) >= 2:
        # 거리 변화 계산
        distance_change = recent_distances[-1] - recent_distances[0]
        change_percent = (distance_change / recent_distances[0]) * 100 if recent_distances[0] > 0 else 0
        
        if change_percent > 10:  # 10% 이상 증가
            return f"고양이가 다가오고 있습니다! (+{change_percent:.1f}%)", "approaching"
        elif change_percent < -10:  # 10% 이상 감소
            return f"고양이가 멀어지고 있습니다! ({change_percent:.1f}%)", "moving_away"
        else:
            return f"고양이가 안정적입니다. (변화: {change_percent:.1f}%)", "stable"
    
    return "분석 중...", "neutral"


#####
    # 웹캠 설정
camera_index = st.selectbox("카메라 선택", [0, 1])

# 거리 추적 설정
st.sidebar.header("고양이 거리 추적 설정")
track_distance = st.sidebar.checkbox("거리 추적 활성화", value=True)
confidence_threshold = st.sidebar.slider("신뢰도 임계값", 0.1, 1.0, 0.5, 0.1)

# 웹캠 시작 버튼
if st.button("웹캠 시작"):
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        st.error("카메라를 열 수 없습니다. 다른 카메라 인덱스를 시도해보세요.")
    else:
        stframe = st.empty()
        distance_info = st.empty()
        movement_info = st.empty()
        stop_button = st.button("웹캠 중지")
        
        # 웹캠 정보
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30
        
        st.info(f"카메라 해상도: {frame_width}x{frame_height}, FPS: {fps}")
        
        # 실시간 처리
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("프레임을 읽을 수 없습니다.")
                break
            
            results = model(frame)
            
            # 탐지 결과를 문자열로 변환
            detection_text = ""
            cat_detected = False
            current_cat_distance = None
            
            # 각 탐지 결과 처리
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    # 클래스별 개수 세기
                    class_counts = {}
                    for box in boxes:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        # 신뢰도가 일정 이상인 것만
                        if confidence > confidence_threshold:
                            if class_name in class_counts:
                                class_counts[class_name] += 1
                            else:
                                class_counts[class_name] = 1
                            
                            # 고양이 거리 추적
                            if track_distance and class_name == "cat":
                                cat_detected = True
                                distance_score, box_area = calculate_cat_distance(box, frame_width, frame_height)
                                current_cat_distance = distance_score
                                
                                # 거리 히스토리에 추가
                                st.session_state.cat_distance_history.append(distance_score)
                                st.session_state.last_cat_detection_time = time.time()
                    
                    # 문자열 형식으로 변환
                    if class_counts:
                        detection_parts = []
                        for class_name, count in class_counts.items():
                            detection_parts.append(f"{count} {class_name}")
                        detection_text = f"0: {frame.shape[1]}x{frame.shape[0]} " + ", ".join(detection_parts)
            
            # 결과 출력
            if detection_text:
                st.write(f"탐지 결과: {detection_text}")
            
            # 고양이 거리 정보 표시
            if cat_detected and current_cat_distance is not None:
                distance_info.success(f"고양이 발견! 거리 점수: {current_cat_distance:.2f}")
                
                # 움직임 분석
                movement_text, movement_status = analyze_cat_movement(st.session_state.cat_distance_history)
                
                if movement_status == "approaching":
                    movement_info.warning(f"{movement_text}")
                elif movement_status == "moving_away":
                    movement_info.info(f"📏 {movement_text}")
                else:
                    movement_info.success(f"{movement_text}")
                    
            elif track_distance:
                # 고양이가 감지되지 않았지만 최근에 감지된 경우
                time_since_last = time.time() - st.session_state.last_cat_detection_time
                if time_since_last < 5:  # 5초 이내에 감지된 경우
                    distance_info.info("고양이를 찾는 중...")
                    movement_info.empty()
                else:
                    distance_info.empty()
                    movement_info.empty()
            
            if detection_text.find("cat") > 0:
                st.success("고양이를 발견했습니다!")
            
            result_frame = results[0].plot()
            stframe.image(result_frame, channels="BGR")
            
            # 대기 
            time.sleep(0.1)
        
        cap.release()
        st.success("웹캠이 중지되었습니다.")
