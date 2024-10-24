from ultralytics import YOLO

import numpy as np


model = YOLO(r'C:\Users\abina\runs\classify\train4\weights\last.pt')  # load a custom model

results = model('raing result.jpg')  # predict on an image

names_dict = results[0].names

probs = results[0].probs.data.tolist()

print(names_dict)
print(probs)

print(names_dict[np.argmax(probs)])