import os
import numpy as np
import torch
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
from MBTR import  create_mobilevit

def main():
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    img_path = "both.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # [C, H, W] -> [N, C, H, W]
    input_tensor = tf.expand_dims(img_tensor, axis=0, name=None)


    model = create_mobilevit(num_classes=7)
    tf.keras.Model.load_weights(filepath="mbvit.h5", by_name=False, skip_mismatch=False, options=None)
    model.layers[-1].activation = None
    target_layers = "last_conv"

    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  use_cuda=True)
    target_category = 0

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()
