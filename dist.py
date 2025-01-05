import cv2
from ultralytics import YOLO, solutions

model = YOLO("yolov8n.pt")
names = model.model.names
from cv2 import VideoCapture
cap = cv2.VideoCapture("C:/Users/yloghmari/Downloads/video1.mp4") # here change the path to ur video.mp4
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter("distance_calculation.avi", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)) # video writer

#initilisation de distance calculation obj
dist_obj = solutions.DistanceCalculation(names=names, view_img=True)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print ("Video frame is empty pr the vid processing has been succssfully completed")
        break

    tracks = model.track(im0, persist=True, show=False)
    im0 = dist_obj.start_process(im0,tracks)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
