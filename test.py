import os
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
import time
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.models as models
import   torchsummary
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def img_transform(img_rgb, transform=None):
    """
    将数据转换为模型读取的形式
    :param img_rgb: PIL Image
    :param transform: torchvision.transform
    :return: tensor
    """

    if transform is None:
        raise ValueError("找不到transform！必须有transform对img进行处理")

    img_t = transform(img_rgb)
    return img_t


def load_class_names(p_clsnames, p_clsnames_cn):
    """
    加载标签名
    :param p_clsnames:
    :param p_clsnames_cn:
    :return:
    """
    with open(p_clsnames, "r") as f:
        class_names = json.load(f)
    with open(p_clsnames_cn, encoding='UTF-8') as f:  # 设置文件对象
        class_names_cn = f.readlines()
    return class_names, class_names_cn


def get_model(path_state_dict, num_classes, vis_model=False):
    """
    创建模型，加载参数
    :param path_state_dict:
    :return:
    """

    model = models.vgg16(num_classes=num_classes)
    model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 100),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(p=0.5),
                                           torch.nn.Linear(100, 2))

    pretrained_state_dict = torch.load(path_state_dict)
    model.load_state_dict(pretrained_state_dict)
    model.eval()

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224), device="cpu")

    model.to(device)
    return model


def process_img(path_img):

    # hard code
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    inference_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # path --> img
    img_rgb = Image.open(path_img).convert('RGB')

    # img --> tensor
    img_tensor = img_transform(img_rgb, inference_transform)
    img_tensor.unsqueeze_(0)        # chw --> bchw
    img_tensor = img_tensor.to(device)

    return img_tensor, img_rgb


if __name__ == "__main__":
    num_classes=2
    # config
    path_state_dict = os.path.join(BASE_DIR, "model.pth")
    path_img = "./data/test/dog/dog.1005.jpg"

    # 1/5 load img
    img_tensor, img_rgb = process_img(path_img)

    # 2/5 load model
    model = get_model(path_state_dict,num_classes, True)

    with torch.no_grad():
        time_tic = time.time()
        outputs = model(img_tensor)
        time_toc = time.time()

    # 4/5 index to class names
    _, pred_int = torch.max(outputs.data, 1)
    _, top1_idx = torch.topk(outputs.data, 1, dim=1)
    #
    pred_idx = int(pred_int.cpu().numpy())
    if pred_idx == 0:
        pred_str= str("cat")
        print("img: {} is: {}".format(os.path.basename(path_img), pred_str))
    else:
        pred_str = str("dog")
        print("img: {} is: {}".format(os.path.basename(path_img), pred_str))
    print("time consuming:{:.2f}s".format(time_toc - time_tic))

    # 5/5 visualization
    plt.imshow(img_rgb)
    plt.title("predict:{}".format(pred_str))
    plt.text(5, 45, "top {}:{}".format(1, pred_str), bbox=dict(fc='yellow'))
    plt.show()
