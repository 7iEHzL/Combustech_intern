import cv2
import numpy as np
from IPython.display import display, clear_output
from PIL import Image as PILImage
import io
import IPython

# YOLO 모델 파일 경로
cfg_file = '"C:/Users/user/Downloads/yolov3.cfg"'
weights_file = 'C:/Users/user/yolov3.weights'
names_file = '"C:/Users/user/Downloads/coco.names"'

# 클래스 이름 불러오기
with open(names_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# YOLO 모델 불러오기
net = cv2.dnn.readNet(weights_file, cfg_file)

# 출력 레이어 가져오기
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# 색상 정의
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# 웹캠 초기화
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    try:
        while True:
            # 프레임 캡처
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # 프레임 크기 가져오기
            height, width, channels = frame.shape

            # 블롭 생성
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # 정보 초기화
            class_ids = []
            confidences = []
            boxes = []

            # 각 감지에서 정보 추출
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # 객체 인식
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # 좌표 계산
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # 비최대 억제 적용
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            # 결과 시각화
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[class_ids[i]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 프레임을 RGB로 변환
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 이미지를 PIL 형식으로 변환
            pil_img = PILImage.fromarray(frame)
            # 이미지를 바이트 형식으로 변환
            with io.BytesIO() as buf:
                pil_img.save(buf, 'jpeg')
                byte_im = buf.getvalue()

            # Jupyter Notebook에 이미지 표시
            clear_output(wait=True)
            display(IPython.display.Image(data=byte_im))

            # 'q' 키를 누르면 루프 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # 웹캠 해제
        cap.release()
        cv2.destroyAllWindows()
