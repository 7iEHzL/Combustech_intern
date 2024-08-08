import cv2
import numpy as np
import selectivesearch
import matplotlib.pyplot as plt

def iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    # Determine the coordinates of the intersection rectangle
    inter_x1 = max(x1, x1_)
    inter_y1 = max(y1, y1_)
    inter_x2 = min(x2, x2_)
    inter_y2 = min(y2, y2_)

    # Compute the area of intersection rectangle
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Compute the area of both bounding boxes
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)

    # Compute the IoU
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def non_max_suppression(boxes, iou_threshold):
    """Perform non-maximum suppression (NMS) on the bounding boxes."""
    if len(boxes) == 0:
        return []

    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep_boxes = []

    while boxes:
        # Select the box with the highest score
        current_box = boxes.pop(0)
        keep_boxes.append(current_box)

        # Remove boxes with IoU greater than the threshold
        boxes = [box for box in boxes if iou(current_box[:4], box[:4]) < iou_threshold]
        
    return keep_boxes

# Initialize camera
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Cannot open the camera.")
    camera.release()
    exit()

while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
        print("Cannot read a frame.")
        break

    # Perform selective search
    img_segments, regions = selectivesearch.selective_search(frame, scale=50, min_size=5000)

    # Filter regions with large size
    large_rects = [region['rect'] for region in regions if region['size'] > 5000]

    # Convert rectangles to bounding boxes with scores
    # Adding a dummy score of 1.0 for NMS purposes
    boxes = [(x, y, x+w, y+h, 1.0) for x, y, w, h in large_rects]

    # Apply Non-Maximum Suppression
    nms_boxes = non_max_suppression(boxes, iou_threshold=0.7)

    # Draw bounding boxes on the frame
    for box in nms_boxes:
        x1, y1, x2, y2, score = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

    # Show the frame with bounding boxes
    cv2.imshow('Frame', frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
camera.release()
cv2.destroyAllWindows()
