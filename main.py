from ultralytics import YOLO

model=YOLO("yolov8n.pt")

print("Hello World")
results=model.train(data="tennis_data.yaml",epochs=3)
