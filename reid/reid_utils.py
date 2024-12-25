import numpy as np
from scipy.spatial.distance import cdist
import cv2
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import deque
import time
from typing import List, Callable, Dict, Tuple
import random

class FeaturePreprocessor:
    """特征预处理器"""
    def __init__(self, normalize=True, remove_outliers=True, smooth=True):
        self.normalize = normalize
        self.remove_outliers = remove_outliers
        self.smooth = smooth
        
    def process(self, features: np.ndarray) -> np.ndarray:
        """预处理特征"""
        if len(features.shape) > 2:
            n_samples = features.shape[0]
            features = features.reshape(n_samples, -1)
            
        processed_features = features.copy()
        
        if self.normalize:
            # L2归一化
            norms = np.linalg.norm(processed_features, axis=1, keepdims=True)
            processed_features = processed_features / (norms + 1e-6)
            
        if self.remove_outliers and len(processed_features) > 2:
            # 使用IQR方法移除离群值
            distances = cdist(processed_features, processed_features.mean(axis=0, keepdims=True)).squeeze()
            q1, q3 = np.percentile(distances, [25, 75])
            iqr = q3 - q1
            mask = (distances >= q1 - 1.5 * iqr) & (distances <= q3 + 1.5 * iqr)
            processed_features = processed_features[mask]
            
        if self.smooth and len(processed_features) > 2:
            # 使用滑动窗口平滑特征
            window_size = min(5, len(processed_features))
            weights = np.ones(window_size) / window_size
            smoothed_features = np.zeros_like(processed_features)
            for i in range(processed_features.shape[1]):
                smoothed_features[:, i] = np.convolve(processed_features[:, i], 
                                                    weights, mode='same')
            processed_features = smoothed_features
            
        return processed_features

class EnhancedFeatureQualityEvaluator:
    """增强的特征质量评估器"""
    def __init__(self):
        self.metrics = {
            'similarity': 0.3,  # 相似度权重
            'stability': 0.3,   # 稳定性权重
            'coverage': 0.2,    # 覆盖度权重
            'density': 0.2      # 密度权重
        }
        
    def evaluate_features(self, features: np.ndarray) -> dict:
        try:
            # 确保特征是2D数组
            if len(features.shape) > 2:
                n_samples = features.shape[0]
                features = features.reshape(n_samples, -1)
                
            # 计算成对距离矩阵, cdist矩阵中值越小越相似
            distances = cdist(features, features, metric='cosine')
            np.fill_diagonal(distances, 1.0)
            
            # 1. 相似度评分
            similarity_score = 1.0 - np.min(distances[distances < 1.0])
            
            # 2. 稳定性评分
            stability_score = 1.0 - np.std(distances)
            
            # 3. 覆盖度评分（特征空间覆盖程度）
            center = np.mean(features, axis=0, keepdims=True)
            mean_dist_to_center = np.mean(cdist(features, center))
            coverage_score = np.clip(mean_dist_to_center / 0.5, 0, 1)
            
            # 4. 密度评分（特征聚集程度）
            k = min(5, len(features)-1)  # k近邻数量
            knn_distances = np.partition(distances, k, axis=1)[:, :k] # 对角线填充为1了
            density_score = 1.0 - np.mean(knn_distances)
            
            # 计算加权总分
            total_score = (
                self.metrics['similarity'] * similarity_score +
                self.metrics['stability'] * stability_score +
                self.metrics['coverage'] * coverage_score +
                self.metrics['density'] * density_score
            )
            
            # 分析问题
            issues = []
            if similarity_score < 0.6:
                issues.append("Low feature similarity")
            if stability_score < 0.6:
                issues.append("Unstable features")
            if coverage_score < 0.4:
                issues.append("Poor space coverage")
            if density_score < 0.5:
                issues.append("Scattered features")
                
            return {
                'total_score': float(total_score),
                'metrics': {
                    'similarity': float(similarity_score),
                    'stability': float(stability_score),
                    'coverage': float(coverage_score),
                    'density': float(density_score)
                },
                'issues': issues,
                'statistics': {
                    'n_samples': len(features),
                    'feature_dim': features.shape[1],
                    'mean_distance': float(np.mean(distances)),
                    'std_distance': float(np.std(distances))
                }
            }
            
        except Exception as e:
            print(f"Error in feature evaluation: {str(e)}")
            return {
                'total_score': 0.0,
                'metrics': {},
                'issues': ['Evaluation failed'],
                'statistics': {}
            }

def calculate_iou(box1, box2):
    """计算两个框的IOU"""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    intersect_x1 = max(x1, x3)
    intersect_y1 = max(y1, y3)
    intersect_x2 = min(x2, x4)
    intersect_y2 = min(y2, y4)
    
    if intersect_x2 < intersect_x1 or intersect_y2 < intersect_y1:
        return 0.0
        
    intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    
    return intersect_area / float(box1_area + box2_area - intersect_area)


class AugmentationCombiner:
    def __init__(self):
        # 定义所有增强方法及其参数范围
        self.augmentation_methods: Dict[str, Tuple[Callable, Dict]] = {
            'brightness': (
                self.adjust_brightness,
                {'factor': (0.7, 1.3)}  # 亮度范围
            ),
            'contrast': (
                self.adjust_contrast,
                {'factor': (0.8, 1.2)}  # 对比度范围
            ),
            'hue_saturation': (
                self.adjust_hue_saturation,
                {
                    'hue_shift': (-10, 10),     # 色调范围
                    'sat_scale': (0.8, 1.2)     # 饱和度范围
                }
            ),
            'noise': (
                self.add_gaussian_noise,
                {
                    'mean': (0, 0),             # 噪声均值范围
                    'std': (5, 25)              # 噪声标准差范围
                }
            ),
            'shadow': (
                self.simulate_shadow,
                {'shadow_level': (0.5, 0.8)}    # 阴影强度范围
            ),
            'random_erasing': (
                self.random_erasing,
                {
                    'sl': (0.02, 0.2),    # 擦除区域面积比例范围
                    'r1': (0.3, 3.3),     # 擦除区域长宽比范围
                    'er_p': (0.5, 0.5)    # 执行擦除的概率
                }
            )
        }

    @staticmethod
    def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * factor, 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
        mean = np.mean(image, axis=(0,1), keepdims=True)
        adjusted = (image - mean) * factor + mean
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    @staticmethod
    def adjust_hue_saturation(image: np.ndarray, hue_shift: float, sat_scale: float) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:,:,0] = (hsv[:,:,0] + hue_shift) % 180
        hsv[:,:,1] = np.clip(hsv[:,:,1] * sat_scale, 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def add_gaussian_noise(image: np.ndarray, mean: float, std: float) -> np.ndarray:
        noise = np.random.normal(mean, std, image.shape)
        noisy_img = image + noise
        return np.clip(noisy_img, 0, 255).astype(np.uint8)

    @staticmethod
    def simulate_shadow(image: np.ndarray, shadow_level: float) -> np.ndarray:
        height, width = image.shape[:2]
        x1, y1 = np.random.randint(0, width), 0
        x2, y2 = np.random.randint(0, width), height
        
        xmin = min(x1, x2)
        xmax = max(x1, x2)
        
        shadow_mask = np.zeros_like(image[:,:,0])
        for y in range(height):
            for x in range(width):
                if xmin <= x <= xmax:
                    shadow_mask[y,x] = 1
                    
        shadow_mask = cv2.GaussianBlur(shadow_mask, (7,7), 3)
        shadow_mask = shadow_mask.reshape(height, width, 1)
        
        result = image.astype(float) * (shadow_level * shadow_mask + (1 - shadow_mask))
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def random_erasing(image: np.ndarray, sl: float = 0.02, sh: float = 0.2, 
                        r1: float = 0.3, er_p: float = 0.5) -> np.ndarray:
        """
        随机擦除增强
        
        参数:
        image: 输入图像
        sl: 擦除区域面积下限（相对于图像面积的比例）
        sh: 擦除区域面积上限（相对于图像面积的比例）
        r1: 擦除区域的长宽比范围
        er_p: 执行擦除的概率
        """
        if random.random() > er_p:  # 按概率执行
            return image
            
        image = image.copy()
        height, width = image.shape[:2]
        img_area = height * width
        
        for _ in range(10):  # 最多尝试10次
            # 随机生成擦除区域的面积和长宽比
            target_area = random.uniform(sl, sh) * img_area
            aspect_ratio = random.uniform(r1, 1/r1)
            
            # 计算擦除区域的宽和高
            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))
            
            # 确保生成的区域不超出图像范围
            if h < height and w < width:
                # 随机选择擦除区域的左上角坐标
                x1 = random.randint(0, width - w)
                y1 = random.randint(0, height - h)
                
                # 生成随机值填充擦除区域
                # 方式1：随机颜色
                if random.random() < 0.5:
                    image[y1:y1+h, x1:x1+w] = np.random.randint(0, 255, size=(h, w, 3))
                # 方式2：图像均值
                else:
                    image[y1:y1+h, x1:x1+w] = np.mean(image, axis=(0,1))
                
                return image
        
        # 如果10次都没有生成合适的区域，返回原图
        return image

    def _get_random_params(self, param_ranges: Dict) -> Dict:
        """生成随机参数"""
        params = {}
        for param_name, (min_val, max_val) in param_ranges.items():
            params[param_name] = random.uniform(min_val, max_val)
        return params

    def augment_image(self, image: np.ndarray, 
                    num_augmentations: int = 3,
                    method_weights: Dict[str, float] = None,
                    quality_threshold: float = None) -> List[np.ndarray]:
        """
        对输入图像进行随机组合增强
        
        参数:
        image: 输入图像
        num_augmentations: 需要生成的增强图像数量
        method_weights: 各种增强方法的选择权重
        quality_threshold: 图像质量阈值（可选）
        
        返回:
        增强后的图像列表
        """
        augmented_images = []
        methods = list(self.augmentation_methods.keys())
        
        # 如果没有提供权重，则使用均匀分布
        if method_weights is None:
            method_weights = {method: 1.0 for method in methods}
        
        weights = [method_weights.get(method, 1.0) for method in methods]
        
        for _ in range(num_augmentations):
            # 随机选择2-3种方法组合
            num_methods = random.randint(2, 3)
            selected_methods = random.choices(methods, 
                                            weights=weights, 
                                            k=num_methods)
            
            # 应用选中的增强方法
            for method in selected_methods:
                aug_image = image.copy()
                func, param_ranges = self.augmentation_methods[method]
                params = self._get_random_params(param_ranges)
                aug_image = func(aug_image, **params)
            
            # 如果设置了质量阈值，进行质量检查
            if quality_threshold is not None:
                quality_score = self.assess_image_quality(aug_image)
                if quality_score < quality_threshold:
                    continue
                
            augmented_images.append(aug_image)
        
        return augmented_images

    def assess_image_quality(self, image: np.ndarray) -> float:
        """
        评估图像质量（示例实现）
        可以根据需求实现更复杂的质量评估方法
        """
        # 计算亮度均值
        brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        
        # 计算对比度（使用标准差作为简单度量）
        contrast = np.std(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        
        # 简单的质量评分（示例）
        # 亮度在中间范围，对比度适中的图像得分较高
        brightness_score = 1.0 - abs(brightness - 127.5) / 127.5
        contrast_score = min(contrast / 50.0, 1.0)
        
        return (brightness_score + contrast_score) / 2.0

# 使用示例
def demo_augmentation():
    # 创建增强器实例
    augmenter = AugmentationCombiner()
    
    # 自定义方法权重（可选）
    method_weights = {
        'brightness': 1.0,
        # 'contrast': 1.0,
        # 'hue_saturation': 0.8,
        'random_erasing': 0.5,  # 添加random_erasing的权重
        'noise': 0.5,
        'shadow': 0.5
    }
    
    # 读取示例图像
    image = cv2.imread('/mnt/c/Users/sunju/Desktop/bak365_video_dataset/sun_gao_identify_cls/rgby_cls_dataset/train/boss/yellow_gao_76.mp4_55_0.jpg')
    
    # 生成增强图像
    augmented_images = augmenter.augment_image(
        image,
        num_augmentations=5,
        method_weights=method_weights,
        quality_threshold=0.6
    )
    
    return augmented_images

if __name__ == '__main__':
    augmented_images = demo_augmentation()
    cv2.imshow('img', np.hstack(augmented_images))
    cv2.waitKey(0)