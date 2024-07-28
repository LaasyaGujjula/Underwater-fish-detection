! pip install ultralytics
! pip install wandb
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from IPython.display import Image, display
!git clone https://github.com/LaasyaGujjula/aquarium_pretrain.git
!pip install opencv-python
!pip install matplotlib
import os
import cv2
import matplotlib.pyplot as plt

# Define paths
root_dir = '/content/aquarium_pretrain'
train_img_path = os.path.join(root_dir, 'train/images/IMG_2274_jpeg_jpg.rf.2f319e949748145fb22dcb52bb325a0c.jpg')
test_img_path = os.path.join(root_dir, 'test/images/IMG_2289_jpeg_jpg.rf.fe2a7a149e7b11f2313f5a7b30386e85.jpg')
# Load images
train_img = cv2.imread(train_img_path)
test_img = cv2.imread(test_img_path)

# Convert BGR to RGB for matplotlib
train_img_rgb = cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB)
test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

# Visualize images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Train Image')
plt.imshow(train_img_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Test Image')
plt.imshow(test_img_rgb)
plt.axis('off')

plt.show()


root_dir = '/content/aquarium_pretrain'
data_yaml_path = os.path.join(root_dir, 'data.yaml')  # Assuming 'data.yaml' is the correct path within the cloned repository


model = YOLO('yolov8n.yaml')
results = model.train(data=data_yaml_path, epochs=100)
from PIL import Image

def detection_pipeline(images):
    output = model(images)

    # Visualize the results
    for i, r in enumerate(output):
        # Plot results image
        im_bgr = r.plot()  # BGR-order numpy array
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # Convert BGR to RGB

        plt.axis('off')
        plt.imshow(im_rgb)
        plt.show()

test_imgs = [
    "/content/aquarium_pretrain/test/images/IMG_2574_jpeg_jpg.rf.ca0c3ad32384309a61e92d9a8bef87b9.jpg",
    "/content/aquarium_pretrain/test/images/IMG_3173_jpeg_jpg.rf.6f05acaa0b22d410a5df3ea3286e227d.jpg",
    "/content/aquarium_pretrain/test/images/IMG_2387_jpeg_jpg.rf.09b38bacfab0922a3a6b66480f01b719.jpg",
    "/content/aquarium_pretrain/test/images/IMG_2434_jpeg_jpg.rf.8b20d3270d4fbc497c64125273f46ecb.jpg",
    "/content/aquarium_pretrain/test/images/IMG_3134_jpeg_jpg.rf.50750ca778773042a3c46a1d3e480132.jpg",
]


detection_pipeline(test_imgs)
