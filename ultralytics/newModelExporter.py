from ultralytics import YOLO



# Create a new YOLO model from scratch
model = YOLO('/content/CustomYoloModelForWasteDetection/ultralytics/cfg/models/v8/yolov8m.yaml')

results = model.train(data='/content/CustomYoloModelForWasteDetection/ultralytics/cfg/datasets/data11.yaml', epochs=100, imgsz=640, batch=16, optimizer='SGD', patience=10)

# Evaluate the model's performance on the validation set
results = model.val()