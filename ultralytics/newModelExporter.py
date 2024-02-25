from ultralytics import YOLO



# Create a new YOLO model from scratch
model = YOLO('/content/CustomYoloModelForWasteDetection/ultralytics/cfg/models/v8/yolov8m.yaml')

results = model.train(data='/content/CustomYoloModelForWasteDetection/ultralytics/cfg/datasets/datasetcust.yaml', epochs=100)

# Evaluate the model's performance on the validation set
results = model.val()

# # Perform object detection on an image using the model

# # Export the model to ONNX format
# # success = model.export(format='pt')
# # model.export(format='pt', weights='best')  # Explicitly export best model



