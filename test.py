from dobot_hci.models.vision import MicrosoftFlorence2
from PIL import Image


model = MicrosoftFlorence2()

img = Image.open("generated_text.jpg")
_, result = model.run_inference(img, "human face")
detections = model.inference_to_sv_detections(result, img)

for detection in detections:
    print(detection[0].tolist(), detections[-1]['class_name'][0])
