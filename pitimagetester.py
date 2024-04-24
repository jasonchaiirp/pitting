from ultralytics import YOLO

# Load a model
model = YOLO("small.pt")  # build a new model from scratch



model.predict('test6.jpg', save=True, show=True, imgsz=320, conf=0.4)
#model.val(plots=True)