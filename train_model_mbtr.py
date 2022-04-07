"""这是使用mbtr算法载入自己的数据集进行训练的脚本"""
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
    # 这里载入GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print()
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    # 这里载入SummaryWriter对象
    # 这边就是我们跑完之后在runs中储存的tensorboard显示路径,建议跑一次就新建一个文件夹,不然出来的图不好看
    tb_writer = SummaryWriter(log_dir="runs/car_experiment")
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    patch_size = 2  # 2x2, for the Transformer blocks.
    image_size = 256
    expansion_factor = 2  # expansion factor for the MobileNetV2 blocks.
    batch_size = 64  #注意这里不要太大,Gpu会爆掉
    num_classes = 7
    epochs = 30   #对于 image training 来说 30 代足够了,不然可能会过拟合

    # 这里就是我说的 train 脚本中的图像处理模块,分为 train 和 val 两个部分
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(256),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(256),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 这段数据就是载入要训练的数据集,这边我直接用的是data-split方法
    #
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # data root path
    image_path = os.path.join(data_root, "--filepath--", "--filename--")  # 加载数据
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # 这边就是加载classes文件了,我自己用的是我自己的车身缺陷分类数据集,后面会在这边直接上传文件的
    defect_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in defect_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=6)  #这个indent是从)开始的,比如你数据集分了7类,你这边就填6
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True,num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = DataLoader(validate_dataset,batch_size=batch_size, shuffle=False,num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    model = MBTR().to(device)

    # 将模型写入tensorboard
    init_img = torch.zeros((1, 3, 256, 256), device=device)
    tb_writer.add_graph(model, init_img)

    # define loss function,这边用的是交叉熵损失,注意不要加载成二分类哦
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer,这边主要是用 Adam分类器,设置 learning_rate
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    best_acc = 0.0

    # 这边就是保存你的训练模型,这边用的是 torch使用较多的 .pth文件形式保存，你自己命名就行
    save_path = './mbtr-1.pth'
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

        # 其实 train_acc 和 val_loss都不是特别重要的参数,你不用的话可以不用刻意添加
        tags = ["train_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], val_accurate, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

    tb_writer.close()
    print('Finished Training')

if __name__=='__main__':
    main()