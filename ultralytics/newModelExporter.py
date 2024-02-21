from ultralytics import YOLO



# Create a new YOLO model from scratch
model = YOLO('/Users/pramudithakarunarathna/Documents/IIT BEng Software Engineering/FYP/Implementations/ultralytics-main/ultralytics/cfg/models/v8/yolov8n.yaml')

results = model.train(data='/Users/pramudithakarunarathna/Documents/IIT BEng Software Engineering/FYP/Implementations/ultralytics-main/ultralytics/cfg/datasets/datasetcust.yaml', epochs=20)

# Evaluate the model's performance on the validation set
results = model.val()

# # Perform object detection on an image using the model

# # Export the model to ONNX format
# # success = model.export(format='pt')
# # model.export(format='pt', weights='best')  # Explicitly export best model



# import torch
# from ultralytics import YOLO

# # 1. Configuration Setup
# config = {
#     'data': '/Users/pramudithakarunarathna/Documents/IIT BEng Software Engineering/FYP/Implementations/ultralytics-main/ultralytics/cfg/datasets/datasetcust.yaml',
#     'model': '/Users/pramudithakarunarathna/Documents/IIT BEng Software Engineering/FYP/Implementations/ultralytics-main/ultralytics/cfg/models/v8/yolov8n.yaml',
#     'epochs': 20,  # Adjust as needed
#     'loss': {
#         'name': 'EIoULoss',
#         'reduction': 'mean'
#     }
# }

# # 2. Model and Datasets
# model = YOLO(config['model']) 
# results = model.train(data=config['data'], epochs=config['epochs'])

# ... (Rest of your evaluation, export code can follow as needed)

