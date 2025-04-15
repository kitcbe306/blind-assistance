# import torch
# import cv2
# import numpy as np

# # Load YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # You can switch to a larger model like yolov5m

# def detect_objects(frame):
#     results = model(frame)  # Run YOLOv5 model
#     detections = results.pandas().xyxy[0]  # Get detections as a DataFrame
    
#     objects = []
#     for _, row in detections.iterrows():
#         obj_name = row['name']
#         conf = row['confidence']
#         xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
#         center_x = (xmin + xmax) / 2
#         center_y = (ymin + ymax) / 2

#         # For simplicity, adding detection info as a tuple
#         objects.append((obj_name, conf, center_x, center_y))
#     return objects


import torch

# Load the YOLOv5 model pre-trained on the COCO dataset
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_objects(frame):
    """
    Detect objects in a given frame using YOLOv5.

    Args:
        frame (numpy.ndarray): The input image frame (BGR format).

    Returns:
        List[Tuple[str, float, int, int]]: A list of detected objects with their names,
                                           confidence scores, and coordinates.
    """
    # Convert the frame from BGR to RGB for YOLOv5
    frame_rgb = frame[:, :, ::-1]

    # Perform inference using the model
    results = model(frame_rgb)

    # Parse the results
    detections = []
    for *box, conf, cls in results.xyxy[0]:  # Iterate through detections
        x_center = int((box[0] + box[2]) / 2)  # X center of the bounding box
        y_center = int((box[1] + box[3]) / 2)  # Y center of the bounding box
        name = model.names[int(cls)]  # Class name
        detections.append((name, float(conf), x_center, y_center))

    return detections
