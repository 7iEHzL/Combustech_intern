from ultralytics import YOLO
import cv2
import numpy as np

# Load a model
model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)

camera = cv2.VideoCapture(0) 

while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
        print("Cannot read a frame.")
        break
    # YOLOv8 모델로 세분화 수행
    results = model.predict(frame)
    boxes = results[0].boxes.xyxy.cpu()
    classes = results[0].boxes.cls.cpu()
    classes_names = results[0].names
    print(classes)
    num = 0
    for i in boxes :
        x_min,y_min,x_max,y_max = map(int,i)
        print(x_min,y_min,x_max,y_max)
        cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),color=(255,0,0),thickness=2)
        cv2.putText(frame,str(classes_names[int(classes[num])]),(x_min,y_min),cv2.FONT_HERSHEY_DUPLEX,1, (0,255,0))
        num += 1
        
    # 세분화 결과 얻기
    segmented_frame = results[0].plot()

    # 결과 프레임 출력
    cv2.imshow('YOLOv8 Segmentation', frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 캡처 객체와 윈도우 해제
camera.release()
cv2.destroyAllWindows()