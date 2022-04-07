"""这是使用mbtr算法载入cifar10数据集进行训练的脚本"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from tqdm import tqdm
import torch.optim as optim
from model_mbtr import MBTR

def main():
    # create model except fc layer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print()
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    # 实例化SummaryWriter对象
    tb_writer = SummaryWriter(log_dir="runs/car_experiment0407-2")
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    patch_size = 2  # 2x2, for the Transformer blocks.
    image_size = 256
    expansion_factor = 2  # expansion factor for the MobileNetV2 blocks.
    batch_size = 128
    num_classes = 10    # 这里是cifar10的种类数,一共10种
    epochs = 50

    transform = transforms.Compose([transforms.RandomResizedCrop(256),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.22050])])
    transform1 = transforms.Compose([transforms.Resize(256),
                               # transforms.CenterCrop(256),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    trainset = datasets.CIFAR10(root='C:/Code/jiang/data_set/cifar-10', train=True,
                                            download=False, transform=transform)
    train_num = len(trainset)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2, pin_memory=True)

    testset = datasets.CIFAR10(root='C:/Code/jiang/data_set/cifar-10', train=False,
                                           download=False, transform=transform1)
    val_num = len(testset)
    validate_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2, pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    model = MBTR().to(device)

    # 将模型写入tensorboard
    init_img = torch.zeros((1, 3, 256, 256), device=device)
    tb_writer.add_graph(model, init_img)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0005)
    best_acc = 0.0
    save_path = './mbtr_cifar10_lr5e-4_ep50.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        model.train()
        running_loss = 0.0
        train_acc = 0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = model(images.to(device))
            pred = torch.max(logits, dim=1)[1]
            train_loss = loss_function(logits, labels.to(device))
            train_loss.backward()
            optimizer.step()

            # print statistics
            running_loss += train_loss.item()

            train_bar.desc = "train epoch[{}/{}] train_acc:{:.3f} loss:{:.3f}".format(epoch + 1,
                                                                                      epochs,
                                                                                      train_loss)

        # validate
        model.eval()
        runval_loss = 0.0
        val_acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                val_loss = loss_function(outputs, val_labels.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                val_acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}] val_acc:{:.3f} val_loss:{:.3f}".format(epoch + 1,
                                                                                      epochs,
                                                                                      val_acc)
        val_accurate = val_acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_path)

        tags = ["train_loss","val_acc","learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], val_accurate, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

    tb_writer.close()
    print('Finished Training')

if __name__=='__main__':
    main()