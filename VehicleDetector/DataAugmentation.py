import os
import random
from PIL import Image, ImageEnhance, ImageOps
import numpy as np


class ImageAugmentor:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder

        # 创建输出文件夹（如果不存在）
        os.makedirs(output_folder, exist_ok=True)

    def _get_image_files(self):
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        return [
            os.path.join(self.input_folder, f)
            for f in os.listdir(self.input_folder)
            if os.path.splitext(f)[1].lower() in supported_formats
        ]

    def rotate(self, image, angle=None):
        if angle is None:
            angle = random.uniform(-45, 45)
        return image.rotate(angle, resample=Image.BICUBIC, expand=True)

    def flip(self, image, method='random'):
        if method == 'random':
            method = random.choice(['horizontal', 'vertical'])

        if method == 'horizontal':
            return ImageOps.mirror(image)
        elif method == 'vertical':
            return ImageOps.flip(image)

    def crop(self, image, scale_range=(0.7, 1.0)):
        width, height = image.size
        scale = random.uniform(*scale_range)

        new_width = int(width * scale)
        new_height = int(height * scale)

        left = random.randint(0, width - new_width)
        top = random.randint(0, height - new_height)

        return image.crop((left, top, left + new_width, top + new_height))

    def adjust_brightness(self, image, factor_range=(0.5, 1.5)):
        enhancer = ImageEnhance.Brightness(image)
        factor = random.uniform(*factor_range)
        return enhancer.enhance(factor)

    def adjust_contrast(self, image, factor_range=(0.5, 1.5)):

        enhancer = ImageEnhance.Contrast(image)
        factor = random.uniform(*factor_range)
        return enhancer.enhance(factor)

    def add_noise(self, image, noise_type='gaussian', intensity=0.05):
        img_array = np.array(image)

        if noise_type == 'gaussian':
            noise = np.random.normal(0, intensity * 255, img_array.shape)
            noisy_image = img_array + noise
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        elif noise_type == 'salt_and_pepper':
            noise_mask = np.random.rand(*img_array.shape[:2]) < intensity
            salt_mask = np.random.rand(*img_array.shape[:2]) < 0.5
            pepper_mask = ~salt_mask

            noisy_image = img_array.copy()
            noisy_image[noise_mask & salt_mask] = 255
            noisy_image[noise_mask & pepper_mask] = 0

        return Image.fromarray(noisy_image)

    def augment(self, num_augmentations=5, augmentation_methods=None):
        if augmentation_methods is None:
            augmentation_methods = [
                self.rotate,
                self.flip,
                self.crop,
                self.adjust_brightness,
                self.adjust_contrast,
                self.add_noise
            ]

        image_files = self._get_image_files()

        for img_path in image_files:
            original_image = Image.open(img_path)

            for i in range(num_augmentations):
                augmented_image = original_image.copy()

                for method in random.sample(augmentation_methods, k=random.randint(1, len(augmentation_methods))):
                    augmented_image = method(augmented_image)

                filename = os.path.splitext(os.path.basename(img_path))[0]
                output_filename = f"{filename}_aug_{i + 1}{os.path.splitext(img_path)[1]}"
                output_path = os.path.join(self.output_folder, output_filename)

                augmented_image.save(output_path)

        print(f"Augmented {len(image_files)} images, exported {len(image_files) * num_augmentations} images")


# 使用示例
def main():
    input_folder = './input_images'
    output_folder = './augmented_images'

    augmentor = ImageAugmentor(input_folder, output_folder)

    # 自定义增强方法（可选）
    custom_methods = [
        augmentor.rotate,
        augmentor.flip,
        augmentor.crop,
        augmentor.adjust_brightness,
        augmentor.adjust_contrast,
        augmentor.add_noise
    ]

    augmentor.augment(num_augmentations=5, augmentation_methods=custom_methods)


if __name__ == "__main__":
    main()