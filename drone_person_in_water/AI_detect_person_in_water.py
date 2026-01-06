import cv2
import torch
from ultralytics import YOLO  # Nếu dùng ultralytics YOLOv5

# Load YOLOv5 model (pre-trained on COCO, includes 'person' class)
model = YOLO('yolov5s.pt')  # Hoặc 'yolov5m.pt' cho chính xác hơn, nhưng chậm hơn

# Hàm detect vùng nước dựa trên màu sắc (xanh dương của biển)
def detect_water(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Range màu xanh dương cho nước biển (có thể điều chỉnh tùy video)
    lower_blue = (90, 50, 50)
    upper_blue = (130, 255, 255)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Làm mịn mask
    mask = cv2.medianBlur(mask, 5)
    return mask  # Trả về binary mask (255: water, 0: non-water)

# Hàm kiểm tra overlap giữa bounding box và vùng nước
def is_in_water(bbox, water_mask, threshold=0.5):
    x1, y1, x2, y2 = map(int, bbox)
    roi = water_mask[y1:y2, x1:x2]
    water_pixels = cv2.countNonZero(roi)
    total_pixels = (x2 - x1) * (y2 - y1)
    return (water_pixels / total_pixels) > threshold if total_pixels > 0 else False

# Đường dẫn video input và output
video_path = 'input_video.mp4'  # Thay bằng đường dẫn video của bạn
output_path = 'output_video.mp4'

# Load video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Writer cho video output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    print(f"Processing frame {frame_count}...")
    
    # Detect objects với YOLOv5
    results = model(frame)
    
    # Lấy detections
    detections = results.xyxy[0]  # [x1, y1, x2, y2, conf, class]
    
    # Detect water mask
    water_mask = detect_water(frame)
    
    for det in detections:
        if int(det[5]) == 0:  # Class 0 là 'person' trong COCO
            conf = det[4]
            if conf > 0.5:  # Threshold confidence
                bbox = det[:4]
                if is_in_water(bbox, water_mask):
                    # Vẽ bounding box xanh cho 'person in water'
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Person in water: {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Lưu frame vào output
    out.write(frame)
    
    # Hiển thị preview (comment nếu không cần)
    # cv2.imshow('Frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Xử lý xong! Video output lưu tại: {output_path}")