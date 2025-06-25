import torch

torch.classes.__path__ = []

import streamlit as st
import cv2
import os
import tempfile
from ultralytics import YOLO

model = YOLO("./yolov8n.pt")
st.title("YOLOv8 실시간 객체 탐지")

# 모드 선택 (웹캠 또는 파일 업로드)
mode = st.radio("입력 모드 선택", ["실시간 웹캠", "비디오 파일 업로드"])

if mode == "실시간 웹캠":
    # 웹캠 설정
    camera_index = st.selectbox("카메라 선택", [0, 1])
    
    # 웹캠 시작 버튼
    if st.button("웹캠 시작"):
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            st.error("카메라를 열 수 없습니다. 다른 카메라 인덱스를 시도해보세요.")
        else:
            stframe = st.empty()
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
                            if confidence > 0.5:  # 임계값 조정 가능
                                if class_name in class_counts:
                                    class_counts[class_name] += 1
                                else:
                                    class_counts[class_name] = 1
                        
                        # 문자열 형식으로 변환
                        if class_counts:
                            detection_parts = []
                            for class_name, count in class_counts.items():
                                detection_parts.append(f"{count} {class_name}")
                            detection_text = f"0: {frame.shape[1]}x{frame.shape[0]} " + ", ".join(detection_parts)
                
                # 결과 출력
                if detection_text:
                    st.write(f"탐지 결과: {detection_text}")
                
                if detection_text.find("cat") > 0:
                    st.success("🐱 고양이를 발견했습니다!")
                
                result_frame = results[0].plot()
                stframe.image(result_frame, channels="BGR")
                
                # 프레임 업데이트를 위한 짧은 대기
                import time
                time.sleep(0.1)
            
            cap.release()
            st.success("웹캠이 중지되었습니다.")

else:
    # 기존 파일 업로드 코드
    video_file = st.file_uploader("비디오 파일 업로드", type=["mp4", "mov", "avi"])

    if video_file:
        # 원본 파일 이름 확보 (확장자 포함)
        filename = video_file.name
        base, ext = os.path.splitext(filename)
        save_dir = "videos"
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, f"{base}_yolo.mp4")

        # 업로드된 파일을 임시 파일로 저장
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        # 비디오 정보 (프레임 크기, FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30

        # 비디오 저장 객체 초기화
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # 프레임 처리
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            # 탐지 결과를 문자열로 변환
            detection_text = ""

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
                        if confidence > 0.5:  # 임계값 조정 가능
                            if class_name in class_counts:
                                class_counts[class_name] += 1
                            else:
                                class_counts[class_name] = 1

                    # 문자열 형식으로 변환
                    if class_counts:
                        detection_parts = []
                        for class_name, count in class_counts.items():
                            detection_parts.append(f"{count} {class_name}")
                        detection_text = f"0: {frame.shape[1]}x{frame.shape[0]} " + ", ".join(detection_parts)

            # 결과 출력
            if detection_text:
                print(detection_text)

            if detection_text.find("cat") > 0:
                print("괭이닷!!!!! 이즈 디스 괭이?")

            result_frame = results[0].plot()
            stframe.image(result_frame, channels="BGR")

            out.write(result_frame)

        cap.release()
        out.release()

        st.success(f"YOLO 결과 비디오가 저장되었습니다: {output_path}")
