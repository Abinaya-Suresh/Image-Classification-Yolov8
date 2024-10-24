from ultralytics import YOLO

model=YOLO('yolov8n-cls.pt')
model.train(data=r'C:\Users\abina\Desktop\Image claasification using yolo\Multi-class Weather Dataset',epochs=20,imgsz=64)