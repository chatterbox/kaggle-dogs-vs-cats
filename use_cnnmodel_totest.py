# coding=utf-8
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, datasets, models
from PIL import Image
from PIL import ImageDraw
import shutil

use_gpu = torch.cuda.is_available()

current_file_num = 0
# 创建模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


net = Net()
if(os.path.exists('model.pt')):
    net=torch.load('model.pt')

base_dir = '/home/zhang/PycharmProjects/test_start/zhang_cat_vs_dog/test/'
sub_dirs = '/home/zhang/PycharmProjects/test_start/zhang_cat_vs_dog/test/test1/'
result_folder = 'result'
path_result = os.path.join(base_dir, result_folder)

for files in os.listdir(base_dir):  # if there is a result file in the root file,delete the old one and create a new one
    if files == result_folder:
        print("the validation folder is already exist")
        shutil.rmtree(path_result, True)

os.makedirs(path_result)


# 对加载的图像作归一化处理， 并裁剪为[224x224x3]大小的图像
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


for files in os.listdir(sub_dirs):
    file_num = len(os.listdir(sub_dirs))
    path_file = os.path.join(sub_dirs, files)
    file_current = Image.open(path_file)
    file_width, file_height = file_current.size
    img_PIL_Tensor = data_transform(file_current)
    if use_gpu:
        images = Variable(img_PIL_Tensor.cuda())
    else:
        images = Variable(img_PIL_Tensor)

    images = images.unsqueeze(0)            # require a 4 dimension tensor to be a input to the cnn
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)

    draw = ImageDraw.Draw(file_current)
    if predicted == 1:
        draw.text((file_width / 2, file_height / 2), "Dog", (0, 0, 0))  # 设置文字位置/内容/颜色/字体
    else:
        draw.text((file_width / 2, file_height / 2), "Cat", (0, 0, 0))  # 设置文字位置/内容/颜色/字体
    draw = ImageDraw.Draw(file_current)
    pathfile_result = os.path.join(path_result, files)
    file_current.save(pathfile_result)
    current_file_num = current_file_num + 1
    print('All files:', file_num, ' Processed files ', current_file_num, 'Files already be done for % .3f' % (current_file_num / file_num) + ' % ' )

