import math
import sys
import os
import time

import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QLabel, QScrollArea, QFrame, QGridLayout, QListWidget,
                             QListWidgetItem, QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QIcon, QPen, QColor
from PyQt5.QtCore import Qt, QMimeData, QSize, QRect


######################
from sklearn.preprocessing import StandardScaler
from PIL import Image
import VehicleDetect
import VehicleNet
import matplotlib.image as mpimg
######################
G_UseMSVM = False

def QPixmapToPILImage(pixmap):
    qimage = pixmap.toImage()
    # QImage -> PIL.Image
    buffer = qimage.bits().asstring(qimage.byteCount())
    image = Image.frombytes("RGBA", (qimage.width(), qimage.height()), buffer)
    if image.mode == "RGBA":
        image = image.convert("RGB")
    return image


def ScaleQRect(rect, scale_factor=2):
    if math.fabs(scale_factor-1) < 0.05:
        return rect
    center = rect.center()  # 获取中心点
    new_width = rect.width() * scale_factor
    new_height = rect.height() * scale_factor
    new_x = center.x() - new_width // 2
    new_y = center.y() - new_height // 2
    return QRect(int(new_x), int(new_y), int(new_width), int(new_height))

def iou(rect1: QRect, rect2: QRect) -> float:
    """ 计算两个 QRect 之间的 IoU（交并比） """
    # 计算交集区域
    if not isinstance(rect1, QRect):
        rect1 = rect1[0]
    if not isinstance(rect2, QRect):
        rect2 = rect2[0]
    x1 = max(rect1.left(), rect2.left())
    y1 = max(rect1.top(), rect2.top())
    x2 = min(rect1.right(), rect2.right())
    y2 = min(rect1.bottom(), rect2.bottom())

    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height

    # 计算各自的面积
    area1 = rect1.width() * rect1.height()
    area2 = rect2.width() * rect2.height()

    # 计算 IoU（交并比）
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0


def non_maximum_suppression(bounding_boxes, confidence_scores, iou_threshold=0.1):
    """
    对边界框执行非极大值抑制

    参数:
        bounding_boxes: QRect对象列表
        confidence_scores: 边界框对应的置信度分数列表
        iou_threshold: IoU阈值，用于决定是否抑制重叠框

    返回:
        保留的边界框索引列表
    """
    # 如果没有边界框，返回空列表
    if len(bounding_boxes) == 0:
        return [], []

    # 按置信度降序排序边界框索引
    indices = sorted(range(len(confidence_scores)), key=lambda i: confidence_scores[i], reverse=True)

    # 保留的边界框索引
    keep_indices = []

    # 按置信度从高到低处理边界框
    while len(indices) > 0:
        # 当前最高置信度的边界框
        current_index = indices[0]
        keep_indices.append(current_index)

        # 比较当前边界框与其他所有边界框
        remaining_indices = []
        for i in indices[1:]:
            # 计算IoU
            iou_v = iou(bounding_boxes[current_index], bounding_boxes[i])

            # 如果IoU小于阈值，保留该边界框
            if iou_v <= iou_threshold:
                remaining_indices.append(i)

        # 更新剩余边界框索引
        indices = remaining_indices

    final_detections = [bounding_boxes[i] for i in keep_indices]
    final_scores = [confidence_scores[i] for i in keep_indices]

    return final_detections, final_scores


class ClickableImageWidget(QLabel):
    def __init__(self, pixmap, box_id = -1, parent=None):
        super().__init__(parent)
        self.original_pixmap = pixmap
        self.setPixmap(pixmap)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid transparent;
            }
            QLabel:hover {
                border: 2px solid blue;
            }
        """)
        self.selected = False
        self.box_id = box_id

    # def mousePressEvent(self, event):
    #     if event.button() == Qt.LeftButton:
    #         # 通知父窗口处理选择事件
    #         if hasattr(self, 'parent_list'):
    #             self.parent_list.on_image_selected(self)

class ThumbnailItem(QWidget):
    def __init__(self, pixmap, text, box_id = -1, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()

        self.image_label = ClickableImageWidget(pixmap, box_id)
        self.text_label = QLabel(text, self)

        layout.addWidget(self.image_label)
        layout.addWidget(self.text_label)
        self.setLayout(layout)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # 通知父窗口处理选择事件
            if hasattr(self, 'parent_list'):
                self.parent_list.on_image_selected(self.image_label)

class DrawableImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_pixmap = None
        self.bounding_boxes = []
        self.selected_box_id = -1

    def set_pixmap(self, pixmap):
        # 存储原始像素图
        self.original_pixmap = pixmap
        self.selected_box_id = -1 # reset selection
        super().setPixmap(pixmap)

    def draw_bounding_boxes(self, bounding_boxes, active_box_id=-1):
        # 如果没有图像，直接返回
        if self.original_pixmap is None:
            return

        print(f'drawing {len(bounding_boxes)} bounding boxes')
        pixmap_with_boxes = QPixmap(self.original_pixmap)
        painter = QPainter(pixmap_with_boxes)

        # 设置画笔样式
        pen = QPen(Qt.red, 3, Qt.SolidLine)
        penSelected = QPen(Qt.blue, 3, Qt.SolidLine)
        painter.setPen(pen)

        # 获取原始图像和当前控件的尺寸
        image_size = self.original_pixmap.size()
        label_size = self.original_pixmap.size() #self.size()

        # 计算缩放比例和偏移
        scale_x = label_size.width() / image_size.width()
        scale_y = label_size.height() / image_size.height()
        scale = min(scale_x, scale_y)

        # 计算居中偏移
        scaled_width = image_size.width() * scale
        scaled_height = image_size.height() * scale
        offset_x = (label_size.width() - scaled_width) / 2
        offset_y = (label_size.height() - scaled_height) / 2

        # 绘制矩形框
        bbox_idx = -1
        for box in bounding_boxes:
            bbox_idx += 1
            # 按比例缩放矩形
            scaled_box = QRect(
                int(box.x() * scale + offset_x),
                int(box.y() * scale + offset_y),
                int(box.width() * scale),
                int(box.height() * scale)
            )
            painter.setPen(pen)
            if bbox_idx == self.selected_box_id:
                painter.setPen(penSelected)
            painter.drawRect(scaled_box)

        painter.end()

        # 设置绘制后的像素图
        super().setPixmap(pixmap_with_boxes)


class DropImageLabel(DrawableImageLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                color: #888;
                font-size: 16px;
            }
        """)
        self.setText("Drag and drop an image to detect vehicles")
        self.setAlignment(Qt.AlignCenter)

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.acceptProposedAction()

    def dropEvent(self, event):
        # 获取拖拽的图片
        mime_data = event.mimeData()
        if mime_data.hasUrls():
            file_path = mime_data.urls()[0].toLocalFile()
            pixmap = QPixmap(file_path)

            # 调整图像大小以适应标签
            scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # 设置图像
            self.set_pixmap(pixmap)
            self.setText("")

            # 触发检测函数（目前为空）
            self.detect_vehicles(file_path)

    # do prediction
    def detect_vehicles(self, image_path):
        test_img = mpimg.imread(image_path)
        tags = []
        start_time = time.time()
        if G_UseMSVM:
            rectangles, scores, tags = VehicleDetect.process_frame_M(test_img, 0.7)
        else:
            rectangles, scores = VehicleDetect.process_frame(test_img, 2)
        end_time = time.time()
        print(len(rectangles), f'boxes found in image in {end_time - start_time} seconds')

        bounding_boxes = []
        for i in range(0, len(rectangles)):
            rect = rectangles[i]
        # for rect in rectangles:
            x = rect[0][0]
            y = rect[0][1]
            w = rect[1][0] - x
            h = rect[1][1] - y
            qrect = QRect(x,y,w,h)
            qrect = ScaleQRect(qrect, 1.5)
            print(f'Adding box {qrect}, confidence: {scores[i]}')
            bounding_boxes.append(qrect)
        if not G_UseMSVM:
            bounding_boxes, scores = non_maximum_suppression(bounding_boxes, scores)
        if hasattr(self, 'main_window'):
            self.main_window.update_detection_results(image_path, bounding_boxes, tags)
            self.draw_bounding_boxes(bounding_boxes)


# 其余代码保持不变，只需将 QLabel 替换为 DrawableImageLabel
class VehicleDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.current_selected_image = None


    def initUI(self):
        # 设置窗口标题和大小
        self.setWindowTitle('VehicleDetector - Course project for COSC444')
        self.setGeometry(100, 100, 1000, 600)

        # 创建主窗口widget
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # 左侧图像显示面板A - 使用新的 DrawableImageLabel
        self.image_label = DropImageLabel()
        self.image_label.main_window = self
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 右侧布局
        right_layout = QVBoxLayout()

        # 图片信息和检测结果Label
        self.info_label = QLabel("Image info：\nDetection results：")
        self.info_label.setFixedWidth(250)
        self.info_label.setWordWrap(True)

        # 检测结果图像显示控件B（滚动区域）
        self.detection_scroll = QScrollArea()
        self.detection_scroll.setFixedWidth(250)
        self.detection_list = QListWidget()
        self.detection_list.setViewMode(QListWidget.IconMode)
        self.detection_list.setMovement(QListWidget.Static)
        self.detection_list.setSpacing(5)
        self.detection_list.setResizeMode(QListWidget.Adjust)
        # 设置单选模式
        self.detection_list.setSelectionMode(QListWidget.SingleSelection)

        self.detection_scroll.setWidget(self.detection_list)
        self.detection_scroll.setWidgetResizable(True)

        # 添加右侧元素到布局
        right_layout.addWidget(self.info_label)
        right_layout.addWidget(self.detection_scroll)

        # 组合左右布局
        main_layout.addWidget(self.image_label, 7)  # 左侧占70%
        main_layout.addLayout(right_layout, 3)  # 右侧占30%

    def update_detection_results(self, image_path, bounding_boxes, tags = []):
        # 更新图片信息Label
        file_info = os.path.basename(image_path)
        file_size = os.path.getsize(image_path) / 1024
        self.info_label.setText(f"Img info：\nName：{file_info}\nSize：{file_size:.2f} KB\nResult：")

        self.detection_list.clear()
        self.current_selected_image = None

        # Add detected vehicle patches
        pixmap = QPixmap(image_path)
        bbox_idx = -1
        for rect in bounding_boxes:
            bbox_idx = bbox_idx + 1
            sub_pixmap = pixmap.copy(rect)
            sub_pixmap = sub_pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            sub_pixmap_pil = QPixmapToPILImage(sub_pixmap)
            if not G_UseMSVM:
                tag, confidence = VehicleNet.load_and_predict_img(sub_pixmap_pil)
                thumbnail_widget = ThumbnailItem(sub_pixmap, f'CNN:{tag}', bbox_idx)
            elif len(tags)>0:
                tag = tags[bbox_idx]
                thumbnail_widget = ThumbnailItem(sub_pixmap, f'MSVM:{tag}', bbox_idx)
            thumbnail_widget.parent_list = self

            # 创建列表项并设置图像
            item = QListWidgetItem()
            item.setSizeHint(QSize(220, 220))  # 设置项目大小
            self.detection_list.addItem(item)
            self.detection_list.setItemWidget(item, thumbnail_widget)


    def on_image_selected(self, selected_widget):
        # 取消之前选中图像的高亮
        if self.current_selected_image:
            self.current_selected_image.setStyleSheet("""
                        QLabel {
                            border: 2px solid transparent;
                        }
                        QLabel:hover {
                            border: 2px solid blue;
                        }
                    """)

        # 高亮当前选中的图像
        selected_widget.setStyleSheet("""
                    QLabel {
                        border: 2px solid red;
                    }
                """)

        # 更新当前选中图像
        self.current_selected_image = selected_widget

        # 调用图像选择的回调函数
        self.on_detection_image_selected(selected_widget.box_id)

    def on_detection_image_selected(self, selected_bbox_id):
        self.image_label.selected_bbox_id = selected_bbox_id
        self.image_label.update()
        pass


def main():
    app = QApplication(sys.argv)
    ex = VehicleDetectionApp()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()