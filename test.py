import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import csv
import time

model = YOLO("yolov8n.pt")

video_path = "abc.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Cannot open video file {video_path}")
    exit()

tracker = Sort()

lane1_ymin, lane1_ymax = 300, 380
lane2_ymin, lane2_ymax = 381, 450
lane3_ymin, lane3_ymax = 451, 520

lane_counts = {1: set(), 2: set(), 3: set()}

with open("vehicle_counts.csv", mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Vehicle_ID", "Lane", "Frame", "Timestamp", "Lane1_Total", "Lane2_Total", "Lane3_Total"])

frame_num = 0
fps = cap.get(cv2.CAP_PROP_FPS)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    results = model(frame, stream=False)
    detections = []

    for box in results[0].boxes.data.tolist():
        x1, y1, x2, y2, score, cls = box
        if int(cls) in [2, 3, 5, 7]:  
            detections.append([x1, y1, x2, y2, score])

    tracked = tracker.update(np.array(detections)) if detections else []

    
    cv2.line(frame, (0, lane1_ymin), (frame.shape[1], lane1_ymin), (255, 0, 0), 2)
    cv2.line(frame, (0, lane1_ymax), (frame.shape[1], lane1_ymax), (255, 0, 0), 2)
    cv2.line(frame, (0, lane2_ymin), (frame.shape[1], lane2_ymin), (0, 255, 0), 2)
    cv2.line(frame, (0, lane2_ymax), (frame.shape[1], lane2_ymax), (0, 255, 0), 2)
    cv2.line(frame, (0, lane3_ymin), (frame.shape[1], lane3_ymin), (0, 0, 255), 2)
    cv2.line(frame, (0, lane3_ymax), (frame.shape[1], lane3_ymax), (0, 0, 255), 2)

    for x1, y1, x2, y2, track_id in tracked:
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        lane_number = None
        if lane1_ymin <= cy <= lane1_ymax:
            lane_number = 1
        elif lane2_ymin <= cy <= lane2_ymax:
            lane_number = 2
        elif lane3_ymin <= cy <= lane3_ymax:
            lane_number = 3

        if lane_number:
            if track_id not in lane_counts[lane_number]:
                lane_counts[lane_number].add(track_id)
                timestamp = frame_num / fps

                
                with open("vehicle_counts.csv", mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        int(track_id),
                        lane_number,
                        frame_num,
                        round(timestamp, 2),
                        len(lane_counts[1]),
                        len(lane_counts[2]),
                        len(lane_counts[3])
                    ])

        
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        cv2.putText(frame, f"ID {int(track_id)}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    
    cv2.putText(frame, f"Lane 1: {len(lane_counts[1])}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Lane 2: {len(lane_counts[2])}", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Lane 3: {len(lane_counts[3])}", (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Traffic Flow", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Final Counts per Lane:")
for lane in lane_counts:
    print(f"Lane {lane}: {len(lane_counts[lane])} vehicles")
