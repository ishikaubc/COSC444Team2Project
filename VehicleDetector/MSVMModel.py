import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import io, color, transform
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
from glob import glob
import time
import SVMModel
from sklearn.metrics import accuracy_score

def load_images_from_folder(folder, categories, img_size=(64, 64)):
    images = []
    labels = []

    for category_id, category in enumerate(categories):
        path = os.path.join(folder, category)
        for img_path in glob(os.path.join(path, "*.jpg")) + glob(os.path.join(path, "*.png")):
            try:
                # 读取图像并转为灰度
                img = io.imread(img_path)
                if img.shape[-1] == 4:  # 处理RGBA图像
                    img = color.rgba2rgb(img)
                if len(img.shape) == 3 and img.shape[-1] > 1:
                    img = color.rgb2gray(img)

                # 调整大小
                img = transform.resize(img, img_size, anti_aliasing=True)

                images.append(img)
                labels.append(category_id)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    return np.array(images), np.array(labels)


def load_images_from_folder_udacity(folder, categories, img_size=(64, 64)):
    images = []
    labels = []

    for category_id, category in enumerate(categories):
        path = os.path.join(folder, category)
        for img_path in glob(os.path.join(path, "*.jpg")) + glob(os.path.join(path, "*.png")):
            try:
                images.append(img_path)
                labels.append(category_id)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    return np.array(images), np.array(labels)


def extract_hog_features(images, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    features = []

    for img in images:
        # 提取HOG特征
        feature = hog(img, orientations=orientations,
                      pixels_per_cell=pixels_per_cell,
                      cells_per_block=cells_per_block,
                      visualize=False,
                      feature_vector=True)
        features.append(feature)

    return np.array(features)

G_categories = ["Non-Vehicles1000", "Sedan", "Semi", "SUV", "Truck"]  # 替换为你的类别名称

def predict_image(image_path, model, scaler, categories, img_size = (64, 64)):
    img = io.imread(image_path)
    if img.shape[-1] == 4:
        img = color.rgba2rgb(img)
    if len(img.shape) == 3 and img.shape[-1] > 1:
        img = color.rgb2gray(img)

    img = transform.resize(img, img_size, anti_aliasing=True)

    feature = hog(img, orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), visualize=False, feature_vector=True)

    feature_scaled = scaler.transform([feature])

    prediction = model.predict(feature_scaled)[0]
    probabilities = model.predict_proba(feature_scaled)[0]

    print(f"预测类别: {categories[prediction]}")
    print("类别概率:")
    for i, category in enumerate(categories):
        print(f"{category}: {probabilities[i]:.4f}")

def main():
    bUseUdacityConfig = True

    # 设置参数
    data_folder = "dataset"  # 包含类别子文件夹的主文件夹
    img_size = (64, 64)
    test_size = 0.2
    random_state = 42

    # 1. 加载图像
    print("正在加载图像...")
    start_time = time.time()
    if bUseUdacityConfig:
        images, labels = load_images_from_folder_udacity(data_folder, G_categories, img_size)
    else:
        images, labels = load_images_from_folder(data_folder, G_categories, img_size)

    print(f"加载完成! 共 {len(images)} 张图像, 耗时 {time.time() - start_time:.2f} 秒")

    # 2. 提取HOG特征
    print("正在提取HOG特征...")
    start_time = time.time()


    if bUseUdacityConfig:
        colorspace = 'YCrCb'  # 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        orient = 11
        pix_per_cell = 16
        cell_per_block = 2
        hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"

        t = time.time()
        features = SVMModel.extract_features(images, cspace=colorspace, orient=orient,
                                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                        hog_channel=hog_channel)
        features = np.array(features)
    else:
        features = extract_hog_features(images)
    print(f"特征提取完成! 特征维度: {features.shape}, 耗时 {time.time() - start_time:.2f} 秒")

    # 3. 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    # 4. 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. 训练SVM分类器
    print("正在训练SVM分类器...")
    start_time = time.time()
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    svm.fit(X_train_scaled, y_train)
    print(f"训练完成! 耗时 {time.time() - start_time:.2f} 秒")

    # 6. 评估模型
    y_pred = svm.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n总体准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=G_categories))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()
    tick_marks = np.arange(len(G_categories))
    plt.xticks(tick_marks, G_categories, rotation=45)
    plt.yticks(tick_marks, G_categories)

    # 在混淆矩阵中标注数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # 7. 保存模型
    import joblib
    joblib.dump(svm, 'models/MSVM_VehicleDetect.pkl')
    joblib.dump(scaler, 'models/MX_Scaler_VehicleDetect.pkl')

    print("模型保存完成!")




if __name__ == "__main__":
    main()