"""这是使用你train出来的.pth文件进行grad cam的热力分布的脚本"""
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
from model_mbtr import MBTR

def main():
    model = MBTR()
    weights_path = "./mbtr_tr_base.pth"
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    # 这里是我用的target_layers的名称,这个在tensorboard的graphs里可以看到
    target_layers = [model.last_conv]
    # 在这里的data_transformer选择的图像处理方法最好和train脚本中的train_data数据处理方法相同
    data_transform = transforms.Compose([transforms.RandomResizedCrop(256),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # 在img_path中载入想要输出热力分布的图片
    img_path = "test.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img = center_crop_img(img, 256)
    img1 = Image.fromarray(img)
    # [C, H, W]
    img_tensor = data_transform(img1)
    # 扩展四维数据
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  use_cuda=False)

    # 这里是cifar10的种类及其对应编号,可以在target_category中修改参数
    # classes: " 0:'plane', 1:'car', 2:'bird', 3:'cat', 4:'deer',
    #            5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck' "

    target_category = 3

    grayscale_cam = cam(input_tensor=input_tensor,
                        target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()
