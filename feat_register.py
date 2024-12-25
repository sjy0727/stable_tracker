import os
import glob
import time
import argparse
from collections import deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from reid.reid_utils import FeaturePreprocessor, EnhancedFeatureQualityEvaluator, AugmentationCombiner, calculate_iou

from reid.reid_onnx import ReIDONNX

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--use_debug_window',
        action='store_true',
    )

    parser.add_argument(
        '--model',
        type=str,
        default='model/bytetrack_s.onnx',
        # default='byte_tracker/model/yolov8s.onnx',
    )

    parser.add_argument(
        '--reid_model',
        type=str,
        default='model/mgn_R50-ibn_dyn.onnx',
    )

    parser.add_argument(
        '--video',
        type=str,
        # default='sample.mp4',
        default='rtsp://admin:zbzn2024@192.168.0.76:554'
        # default='twitter_x264.mp4',
        # default='/Users/sunjunyi/Downloads/bak.mp4'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output',
    )
    parser.add_argument(
        '--score_th',
        type=float,
        default=0.1,
    )
    parser.add_argument(
        '--nms_th',
        type=float,
        default=0.7,
    )
    parser.add_argument(
        '--input_shape',
        type=str,
        default='608,1088',
    )
    parser.add_argument(
        '--reid_input_shape',
        type=str,
        default='384,128',
    )
    parser.add_argument(
        '--with_p6',
        action='store_true',
        help='Whether your model uses p6 in FPN/PAN.',
    )

    # tracking args
    parser.add_argument(
        '--track_thresh',
        type=float,
        default=0.4,
        help='tracking confidence threshold',
    )
    parser.add_argument(
        '--track_buffer',
        type=int,
        default=30,
        help='the frames for keep lost tracks',
    )
    parser.add_argument(
        '--match_thresh',
        type=float,
        default=0.8,
        help='matching threshold for tracking',
    )
    parser.add_argument(
        '--min-box-area',
        type=float,
        default=10,
        help='filter out tiny boxes',
    )
    parser.add_argument(
        '--mot20',
        dest='mot20',
        default=False,
        action='store_true',
        help='test mot20.',
    )

    args = parser.parse_args()

    return args

class PersonRegister:
    def __init__(self, 
                camera_url: str, 
                save_dir: str = './person_features',
                time_window: float = 3.0,
                min_frames: int = 10,
                iou_threshold: float = 0.5,
                quality_threshold: float = 0.6
                ):
        self.camera_url = camera_url
        self.save_dir = save_dir
        self.time_window = time_window
        self.min_frames = min_frames
        self.iou_threshold = iou_threshold
        self.quality_threshold = quality_threshold
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化模型
        self.det_model = YOLO('model/yolov8n.pt')
        # args = get_parser().parse_args()
        # cfg = setup_cfg(args)
        # self.reid_model = FeatureExtractionDemo(cfg, parallel=args.parallel)
        args = get_args()
        self.reid_model = ReIDONNX(args) # FIXME: 需要传参args
        
        # 特征处理工具
        self.preprocessor = FeaturePreprocessor()
        self.quality_evaluator = EnhancedFeatureQualityEvaluator()
        self.augmentation_combiner = AugmentationCombiner()
        self.method_weights = {
            'brightness': 1.0,
            # 'contrast': 1.0,
            # 'hue_saturation': 0.8,
            'random_erasing': 0.5,  # 添加random_erasing的权重
            'noise': 0.5,
            'shadow': 0.5
        }

        # 状态变量
        self.is_collecting = False
        self.collecting_box = None
        self.feature_buffer = deque(maxlen=500)
        self.collection_start_time = None
        self.current_frame = None
        self.current_detections = []
        
        # 加载已注册ID
        self.registered_ids = self._load_registered_ids()
        
        # 显示配置
        self.display_config = {
            'main_window': "Person Registration",
            'font': cv2.FONT_HERSHEY_SIMPLEX,
            'colors': {
                'text': (255, 255, 255),
                'warning': (0, 0, 255),
                'success': (0, 255, 0),
                'info': (255, 255, 0)
            }
        }
    
    def _load_registered_ids(self):
        """加载已注册的ID"""
        files = glob.glob(f"{self.save_dir}/*.npz")
        ids = [int(f.split('_')[-1].split('.')[0]) for f in files]
        return set(ids) if ids else set()
    
    def _get_next_id(self):
        """获取下一个可用ID"""
        if not self.registered_ids:
            return 1
        return max(self.registered_ids) + 1
    
    def _extract_feature(self, person_img):
        """提取特征"""
        with torch.no_grad():
            # feature = self.reid_model.run_on_image(person_img) # FIXME: 替换为onnx模型
            # feature = postprocess(feature)
            feature = self.reid_model.inference(person_img)[None]
            return feature

    def _save_feature(self, feature, quality_info):
        """保存特征"""
        next_id = self._get_next_id()
        save_path = f"{self.save_dir}/person_{next_id}.npz"
        np.savez(save_path, 
                feature=feature,
                quality_info=quality_info)
        self.registered_ids.add(next_id)
        return next_id

    def _draw_status_panel(self, frame, info=None):
        """绘制状态面板"""
        h, w = frame.shape[:2]
        panel_h = 150
        panel_w = 300
        
        # 创建半透明覆盖层
        overlay = frame.copy()
        cv2.rectangle(overlay, (w-panel_w, 0), (w, panel_h), 
                    (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # 基本信息
        y_offset = 30
        cv2.putText(frame, f"Registered IDs: {len(self.registered_ids)}", 
                    (w-panel_w+10, y_offset),
                    self.display_config['font'], 0.6, 
                    self.display_config['colors']['info'], 2)
        
        if info and 'metrics' in info:
            metrics = info['metrics']
            y_offset += 25
            for metric, value in metrics.items():
                cv2.putText(frame, 
                            f"{metric.capitalize()}: {value:.2f}",
                            (w-panel_w+10, y_offset),
                            self.display_config['font'], 0.6,
                            self._get_score_color(value), 1)
                y_offset += 20
                
        return frame

    def _get_score_color(self, score):
        """根据分数返回颜色"""
        if score >= 0.7:
            return self.display_config['colors']['success']
        elif score >= 0.5:
            return self.display_config['colors']['info']
        else:
            return self.display_config['colors']['warning']

    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.is_collecting:
                print("Already collecting features, please wait...")
                return
                
            # 检查点击位置是否在检测框内
            for det in self.current_detections:
                x1, y1, x2, y2 = det['bbox']
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.is_collecting = True
                    self.collecting_box = det['bbox']
                    self.feature_buffer.clear()
                    self.collection_start_time = time.time()
                    print("Started collecting features...")
                    break

    def start_register(self):
        """启动注册流程"""
        cap = cv2.VideoCapture(self.camera_url)
        if not cap.isOpened():
            print("Failed to open camera")
            return
            
        cv2.namedWindow(self.display_config['main_window'])
        cv2.setMouseCallback(self.display_config['main_window'], self._mouse_callback)
        
        print("Click on a person to start registration. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            self.current_frame = frame.copy()
            frame_display = frame.copy()
            h, w = frame.shape[:2]
            
            # 目标检测
            results = self.det_model(frame, classes=0, verbose=False)
            self.current_detections = []
            
            if len(results[0].boxes):
                for r in results[0].boxes.data:
                    x1, y1, x2, y2, conf, cls = r
                    if conf < 0.5:
                        continue
                    
                    bbox = (int(x1), int(y1), int(x2), int(y2))
                    self.current_detections.append({
                        'bbox': bbox,
                        'conf': conf
                    })
                    
                    # 如果正在收集特征
                    if self.is_collecting:
                        if calculate_iou(bbox, self.collecting_box) > self.iou_threshold:
                            # 提取并存储特征
                            person_img = frame[int(y1):int(y2), int(x1):int(x2)]
                            downsampled_person_img = cv2.pyrDown(person_img)
                            augumented_person_imgs = self.augmentation_combiner.augment_image(
                                person_img,
                                num_augmentations=4,
                                method_weights=self.method_weights,
                                quality_threshold=0.6)
                            feature = self._extract_feature(person_img)
                            downsampled_feature = self._extract_feature(downsampled_person_img)
                            augumented_features = [self._extract_feature(img) for img in augumented_person_imgs]
                            self.feature_buffer.append(feature)
                            self.feature_buffer.append(downsampled_feature)
                            self.feature_buffer.extend(augumented_features)
                            
                            # 使用红色框标注正在收集的目标
                            cv2.rectangle(frame_display, (int(x1), int(y1)), 
                                        (int(x2), int(y2)), (0, 0, 255), 2)
                            continue
                    
                    # 普通检测框使用绿色
                    cv2.rectangle(frame_display, (int(x1), int(y1)), 
                                (int(x2), int(y2)), (0, 255, 0), 2)
                    # 显示提示文字
                    cv2.putText(frame_display, 
                                "Click to register", 
                                (int(x1), int(y1)-10),
                                self.display_config['font'], 1.0,
                                (0, 255, 0), 2)
            
            # 检查是否正在收集特征
            if self.is_collecting:
                elapsed_time = time.time() - self.collection_start_time
                remaining_time = max(0, self.time_window - elapsed_time)
                
                # 显示收集进度
                cv2.putText(frame_display,
                            f"Collecting: {len(self.feature_buffer)}/{self.min_frames} frames, {remaining_time:.1f}s left", 
                            (10, 60),
                            self.display_config['font'], 1, 
                            (0, 0, 255), 2)
                
                # 检查是否完成收集
                if (elapsed_time >= self.time_window and 
                    len(self.feature_buffer) >= self.min_frames):
                    
                    try:
                        # 处理收集到的特征
                        features = np.stack([f for f in self.feature_buffer])
                        print(f"Raw features shape: {features.shape}")
                        
                        # 预处理特征
                        processed_features = self.preprocessor.process(features)
                        print(f"Processed features shape: {processed_features.shape}")
                        
                        # 质量评估
                        quality_info = self.quality_evaluator.evaluate_features(processed_features)
                        
                        # 更新显示
                        frame_display = self._draw_status_panel(frame_display, quality_info)
                        
                        if quality_info['total_score'] >= self.quality_threshold:
                            # 计算最终特征
                            avg_feature = np.mean(processed_features, axis=0)
                            person_id = self._save_feature(avg_feature, quality_info)
                            
                            # 显示成功信息
                            success_text = f"Success! ID: {person_id} Score: {quality_info['total_score']:.2f}"
                            cv2.putText(frame_display, success_text,
                                        (10, h-30),
                                        self.display_config['font'], 1,
                                        self.display_config['colors']['success'], 2)
                            
                            print(f"Registration complete! Person ID: {person_id}")
                            print(f"Quality score: {quality_info['total_score']:.2f}")
                            print("Metrics:", quality_info['metrics'])
                            
                        else:
                            # 显示失败原因
                            issues_text = " | ".join(quality_info['issues'])
                            cv2.putText(frame_display,
                                        f"Failed: {issues_text}",
                                        (10, h-30),
                                        self.display_config['font'], 1,
                                        self.display_config['colors']['warning'], 2)
                            
                            print("Registration failed due to low quality")
                            print("Issues:", issues_text)
                            
                    except Exception as e:
                        print(f"Error processing features: {str(e)}")
                        cv2.putText(frame_display,
                                    "Error processing features",
                                    (10, h-30),
                                    self.display_config['font'], 1,
                                    self.display_config['colors']['warning'], 2)
                    
                    # 重置收集状态
                    self.is_collecting = False
                    self.collecting_box = None
                    self.feature_buffer.clear()
            
            # 显示基本信息
            cv2.putText(frame_display,
                        f"Registered IDs: {len(self.registered_ids)}", 
                        (10, 30),
                        self.display_config['font'], 1, 
                        self.display_config['colors']['info'], 2)
            
            # 显示操作说明
            help_text = "Click person to register | Press 'q' to quit"
            cv2.putText(frame_display, help_text,
                        (10, h-60),
                        self.display_config['font'], 1.0,
                        self.display_config['colors']['text'], 2)
            
            # 显示结果
            cv2.imshow(self.display_config['main_window'], frame_display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 清理资源
        cap.release()
        cv2.destroyAllWindows()

def main():
    camera_url = "rtsp://admin:zbzn2024@192.168.0.77:554"  # 替换为实际的相机URL
    register = PersonRegister(
        camera_url=camera_url,
        time_window=10.0,    # 10秒时间窗口
        min_frames=20,      # 最少需要20帧
        iou_threshold=0.5,  # IOU阈值0.5
        quality_threshold=0.6  # 质量阈值0.6
    )
    
    try:
        register.start_register()
    except KeyboardInterrupt:
        print("\nRegistration interrupted by user")
    except Exception as e:
        print(f"Error during registration: {str(e)}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()