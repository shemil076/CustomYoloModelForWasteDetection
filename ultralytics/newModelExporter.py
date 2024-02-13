from ultralytics import YOLO

import importlib.util
spec = importlib.util.spec_from_file_location("ChannelAttentionModule", "../ultralytics/models/experimental/attention.py")
attention_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(attention_module)


# Create a new YOLO model from scratch
model = YOLO('/Users/pramudithakarunarathna/Documents/IIT BEng Software Engineering/FYP/Implementations/ultralytics-main/ultralytics/cfg/models/v8/yolov8n.yaml')

results = model.train(data='/Users/pramudithakarunarathna/Documents/IIT BEng Software Engineering/FYP/Implementations/ultralytics-main/ultralytics/cfg/datasets/datasetcust.yaml', epochs=3)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model

# Export the model to ONNX format
# success = model.export(format='pt')
# model.export(format='pt', weights='best')  # Explicitly export best model
