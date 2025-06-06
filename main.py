# -*- coding: utf-8 -*-

"""
Projeto EyeGrid - Cidadela SOS
Detector de Gestos de Ajuda com Python e MediaPipe

Este script utiliza a webcam ou um arquivo de vídeo para identificar pessoas
acenando em um gesto de pedido de ajuda. A detecção é feita analisando a posição
dos pulsos em relação aos ombros.

Quando um gesto de aceno é detectado por um período contínuo, um alerta
visual é exibido na tela.

Bibliotecas necessárias:
- opencv-python
- mediapipe

Para instalar, use o comando:
pip install opencv-python mediapipe
"""

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_ALERT = (0, 0, 255)
COLOR_NORMAL = (0, 255, 0)
TEXT_STROKE = 2

WAVE_FRAMES_THRESHOLD = 15

def main():
    """
    Função principal que executa a captura de vídeo e a detecção de gestos.
    """
    cap = cv2.VideoCapture("video.mp4")

    wave_counter = 0
    alert_triggered = False

    with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Não foi possível carregar o frame. Fim do vídeo ou erro na câmera.")
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            results = pose.process(image_rgb)

            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

                is_waving = (left_wrist.y < left_shoulder.y) or \
                            (right_wrist.y < right_shoulder.y)

                if is_waving:
                    wave_counter += 1
                    if wave_counter > WAVE_FRAMES_THRESHOLD:
                        alert_triggered = True
                else:
                    wave_counter = 0
                    alert_triggered = False

            except (AttributeError, IndexError):
                wave_counter = 0
                alert_triggered = False
                pass

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image_bgr,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

            if alert_triggered:
                status_text = "ALERTA: Gesto de AJUDA detectado!"
                text_color = COLOR_ALERT
            else:
                status_text = "Status: Monitorando..."
                text_color = COLOR_NORMAL

            cv2.rectangle(image_bgr, (0, 0), (image_bgr.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(image_bgr, status_text, (10, 30), FONT, 1, text_color, TEXT_STROKE)

            cv2.imshow('EyeGrid - Cidadela SOS', image_bgr)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
