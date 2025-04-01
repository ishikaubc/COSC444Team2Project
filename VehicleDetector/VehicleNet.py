import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime, time


G_Classes = ['Sedan', 'Truck', 'SUV', 'Semi', 'Non-Vehicles']
class CarDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = G_Classes  # 包括非汽车类别
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: {class_dir} directory not found!")
                continue
            for filename in os.listdir(class_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(class_dir, filename), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个随机生成的图像和标签，以避免中断训练过程
            dummy_img = torch.randn(3, 64, 64)
            return dummy_img, label

# 定义CNN模型
class CarClassifier(nn.Module):
    def __init__(self, num_classes=len(G_Classes)):
        super(CarClassifier, self).__init__()

        # 使用预训练的ResNet18作为特征提取器
        self.model = models.resnet18(pretrained=True)

        # 替换最后的全连接层
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# 设置数据变换
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

# 预测新图像函数
def predict_image(model, image_path, device):
    model = model.to(device)
    model.eval()

    class_names = G_Classes

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]

        # 输出所有类别的概率
        for i, prob in enumerate(probs):
            print(f"{class_names[i]}: {prob.item():.4f}")

        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()], probs[predicted.item()].item()

# 批量预测文件夹中的图像并保存结果
def predict_folder(model, folder_path, device, save_results=True, output_dir='.'):
    import glob

    model.to(device)
    model.eval()

    # 获取所有图像文件
    image_files = []
    for ext in ['jpg', 'jpeg', 'png']:
        image_files.extend(glob.glob(os.path.join(folder_path, f'*.{ext}')))

    results = []

    for img_path in image_files:
        try:
            class_name, confidence = predict_image(model, img_path, device)
            print(f"Image: {os.path.basename(img_path)}")
            print(f"Predicted: {class_name} with confidence: {confidence:.4f}")
            print("-" * 30)

            results.append({
                'image': os.path.basename(img_path),
                'predicted_class': class_name,
                'confidence': confidence
            })
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    if save_results and results:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(output_dir, f'batch_predictions_{timestamp}.json')

        import json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"Results saved to {results_path}")

    return results


# 加载已训练的模型并进行预测
G_Model_Lib = {}
# best_model_path = 'models/car_classifier_best_20250330_044911.pth'
best_model_path = 'models/car_classifier_best_20250401_095533.pth'


G_Predict_Stat = {"totalTime":0,"totalCount":0.0}
def load_and_predict_img(img, model_path=best_model_path, device=None):
    global G_Model_Lib
    global G_Predict_Stat
    start_time = time.time()
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model_path not in G_Model_Lib:
        print(f"Model from {model_path} unavailable, loading it now...")
        model = CarClassifier(num_classes=len(G_Classes))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        G_Model_Lib[model_path] = model
        print(f"Model loaded from {model_path}")
    else:
        model = G_Model_Lib[model_path]

    model = model.to(device)
    model.eval()

    class_names = G_Classes

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # image = Image.fromarray((img * 255).astype('uint8'))  # 归一化数据需要转换回 [0,255]
    # image = Image.open(image_path).convert('RGB')
    image_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]

        # 输出所有类别的概率
        for i, prob in enumerate(probs):
            print(f"{class_names[i]}: {prob.item():.4f}")

        _, predicted = torch.max(outputs, 1)
        end_time = time.time()
        seconds = end_time - start_time
        G_Predict_Stat['totalTime'] += seconds
        G_Predict_Stat['totalCount'] += 1
        print(f'[VehicleNet]Elapsed time: {seconds:.2f} seconds, average time {G_Predict_Stat["totalTime"]/G_Predict_Stat["totalCount"]}, total inference count: {G_Predict_Stat["totalCount"]}')
        return class_names[predicted.item()], probs[predicted.item()].item()

def load_and_predict(model_path, test_image_path=None, test_folder_path=None):
    global G_Model_Lib
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model_path not in G_Model_Lib:
        print(f"Model from {model_path} unavailable, loading it now...")
        model = CarClassifier(num_classes=len(G_Classes))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        G_Model_Lib[model_path] = model
        print(f"Model loaded from {model_path}")
    else:
        model = G_Model_Lib[model_path]
    # 预测单张图像
    if test_image_path:
        class_name, confidence = predict_image(model, test_image_path, device)
        print(f"Image: {test_image_path}")
        print(f"Predicted: {class_name} with confidence: {confidence:.4f}")

    # 预测文件夹中的所有图像
    if test_folder_path:
        print(f"Processing all images in {test_folder_path}")
        predict_folder(model, test_folder_path, device)

# load_and_predict(best_model_path, test_image_path=None, test_folder_path='test_imgs')
