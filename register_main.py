import cv2
import argparse
import numpy as np
import multiprocessing as mp
import time
import signal
import copy
import sys
import queue
import os
import glob
from typing import Optional, Dict, List
from collections import deque
from multiprocessing.synchronize import Event

import torch
from ultralytics import YOLO

from reid.reid_onnx import ReIDONNX
from reid.reid_utils import (
    FeaturePreprocessor,
    EnhancedFeatureQualityEvaluator,
    AugmentationCombiner,
    calculate_iou
)

from byte_tracker.tracker.byte_tracker import load_registered_features

class SharedState:
    """共享状态类，用于在进程间共享注册相关的状态"""
    def __init__(self):
        self.is_collecting = mp.Value('b', False)
        self.collection_start_time = mp.Value('d', 0.0)
        self.collecting_box = mp.Array('d', [0.0, 0.0, 0.0, 0.0])
        self.feature_queue = mp.Queue(maxsize=500)
        # self.registered_ids = mp.Manager().list()
        self.registered_ids = load_registered_features('person_features')[1]

class VideoStreamProducer(mp.Process):
    def __init__(self, url: str, frame_queue: mp.Queue, control_event: Event, max_queue_size: int = 30):
        super().__init__()
        self.url = url
        self.frame_queue = frame_queue
        self.control_event = control_event
        self.max_queue_size = max_queue_size
        
    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        
        cap: Optional[cv2.VideoCapture] = None
        retry_count = 0
        max_retries = 3
        
        while not self.control_event.is_set():
            try:
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(self.url)
                    if not cap.isOpened():
                        raise ConnectionError(f"Cannot connect to video stream: {self.url}")
                
                ret, frame = cap.read()
                if not ret or frame is None:
                    raise ConnectionError("Failed to read video frame")
                    
                frame_data = {
                    'frame': frame,
                    'timestamp': time.time()
                }
                
                if not self.frame_queue.full():
                    self.frame_queue.put(frame_data, block=False)
                retry_count = 0
                
            except (queue.Full, BrokenPipeError):
                time.sleep(0.01)
                continue
                
            except Exception as e:
                print(f"Producer error: {str(e)}")
                retry_count += 1
                
                if retry_count >= max_retries:
                    print("Max retries reached, exiting producer process")
                    break
                    
                if cap is not None:
                    cap.release()
                    cap = None
                    
                time.sleep(1)
        
        if cap is not None:
            cap.release()

class PersonRegisterConsumer(mp.Process):
    def __init__(
        self,
        frame_queue: mp.Queue,
        control_event: Event,
        shared_state: SharedState,
        save_dir: str = './person_features',
        display_name: str = "Person Registration",
        time_window: float = 3.0,
        min_frames: int = 10,
        iou_threshold: float = 0.5,
        quality_threshold: float = 0.6
    ):
        super().__init__()
        self.frame_queue = frame_queue
        self.control_event = control_event
        self.shared_state = shared_state
        self.save_dir = save_dir
        self.display_name = display_name
        self.time_window = time_window
        self.min_frames = min_frames
        self.iou_threshold = iou_threshold
        self.quality_threshold = quality_threshold
        
        self.display_config = {
            'font': cv2.FONT_HERSHEY_SIMPLEX,
            'colors': {
                'text': (255, 255, 255),
                'warning': (0, 0, 255),
                'success': (0, 255, 0),
                'info': (255, 255, 0)
            }
        }

    def initialize_models(self):
        """Initialize detection and ReID models"""
        self.det_model = YOLO('model/yolov8n.pt')
        self.reid_model = ReIDONNX(get_args())
        self.preprocessor = FeaturePreprocessor()
        self.quality_evaluator = EnhancedFeatureQualityEvaluator()
        self.augmentation_combiner = AugmentationCombiner()
        self.method_weights = {
            'brightness': 1.0,
            'random_erasing': 0.5,
            'noise': 0.5,
            'shadow': 0.5
        }

    def _extract_feature(self, person_img):
        """Extract ReID features from person image"""
        with torch.no_grad():
            feature = self.reid_model.inference(person_img)[None]
            return feature

    def _save_feature(self, feature, quality_info):
        """Save extracted features"""
        next_id = max(self.shared_state.registered_ids) + 1 if self.shared_state.registered_ids else 1
        save_path = f"{self.save_dir}/person_{next_id}.npz"
        np.savez(save_path, feature=feature, quality_info=quality_info)
        self.shared_state.registered_ids.append(next_id)
        return next_id

    def _draw_status_panel(self, frame, info=None):
        """Draw status information panel on frame"""
        h, w = frame.shape[:2]
        panel_h = 150
        panel_w = 300
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (w-panel_w, 0), (w, panel_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        y_offset = 30
        cv2.putText(
            frame, 
            f"Registered IDs: {len(self.shared_state.registered_ids)}", 
            (w-panel_w+10, y_offset),
            self.display_config['font'], 
            0.6, 
            self.display_config['colors']['info'], 
            2
        )
        
        if info and 'metrics' in info:
            metrics = info['metrics']
            y_offset += 25
            for metric, value in metrics.items():
                color = (
                    self.display_config['colors']['success'] if value >= 0.7
                    else self.display_config['colors']['info'] if value >= 0.5
                    else self.display_config['colors']['warning']
                )
                cv2.putText(
                    frame, 
                    f"{metric.capitalize()}: {value:.2f}",
                    (w-panel_w+10, y_offset),
                    self.display_config['font'], 
                    0.6,
                    color, 
                    1
                )
                y_offset += 20
                
        return frame

    def _mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for selecting person to register"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.shared_state.is_collecting.value:
                print("Already collecting features, please wait...")
                return
                
            for det in self.current_detections:
                x1, y1, x2, y2 = det['bbox']
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.shared_state.is_collecting.value = True
                    self.shared_state.collecting_box[:] = det['bbox']
                    self.shared_state.collection_start_time.value = time.time()
                    while not self.shared_state.feature_queue.empty():
                        try:
                            self.shared_state.feature_queue.get_nowait()
                        except:
                            pass
                    print("Started collecting features...")
                    break

    def run(self):
        """Main consumer process loop"""
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        
        # Initialize models and create save directory
        self.initialize_models()
        os.makedirs(self.save_dir, exist_ok=True)
        
        window_created = False
        self.current_detections = []
        frame_id = 0
        
        print("Click on a person to start registration. Press 'q' to quit.")
        
        while not self.control_event.is_set():
            try:
                if self.frame_queue.empty():
                    time.sleep(0.01)
                    continue
                
                frame_data = self.frame_queue.get(block=False)
                frame = frame_data['frame']
                frame_display = frame.copy()
                h, w = frame.shape[:2]
                
                # Create or recreate window
                if not window_created:
                    cv2.namedWindow(self.display_name)
                    cv2.setMouseCallback(self.display_name, self._mouse_callback)
                    window_created = True
                
                # Object detection
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
                        
                        # Process detection based on collection state
                        if self.shared_state.is_collecting.value:
                            current_box = tuple(self.shared_state.collecting_box[:])
                            if calculate_iou(bbox, current_box) > self.iou_threshold:
                                # Extract and store features
                                person_img = frame[int(y1):int(y2), int(x1):int(x2)]
                                
                                # Generate augmented images
                                downsampled_person_img = cv2.pyrDown(person_img)
                                augmented_person_imgs = self.augmentation_combiner.augment_image(
                                    person_img,
                                    num_augmentations=4,
                                    method_weights=self.method_weights,
                                    quality_threshold=0.6
                                )
                                
                                # Extract features from all versions
                                feature = self._extract_feature(person_img)
                                downsampled_feature = self._extract_feature(downsampled_person_img)
                                augmented_features = [
                                    self._extract_feature(img) for img in augmented_person_imgs
                                ]
                                
                                # Store features
                                if not self.shared_state.feature_queue.full():
                                    self.shared_state.feature_queue.put(feature)
                                    self.shared_state.feature_queue.put(downsampled_feature)
                                    for feat in augmented_features:
                                        self.shared_state.feature_queue.put(feat)
                                
                                # Draw red box for collection target
                                cv2.rectangle(
                                    frame_display, 
                                    (int(x1), int(y1)), 
                                    (int(x2), int(y2)), 
                                    (0, 0, 255), 
                                    2
                                )
                                continue
                        
                        # Draw green box for other detections
                        cv2.rectangle(
                            frame_display, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            (0, 255, 0), 
                            2
                        )
                        cv2.putText(
                            frame_display, 
                            "Click to register", 
                            (int(x1), int(y1)-10),
                            self.display_config['font'], 
                            1.0,
                            (0, 255, 0), 
                            2
                        )
                
                # Handle feature collection state
                if self.shared_state.is_collecting.value:
                    elapsed_time = time.time() - self.shared_state.collection_start_time.value
                    remaining_time = max(0, self.time_window - elapsed_time)
                    feature_count = self.shared_state.feature_queue.qsize()
                    
                    cv2.putText(
                        frame_display,
                        f"Collecting: {feature_count}/{self.min_frames} frames, {remaining_time:.1f}s left", 
                        (10, 60),
                        self.display_config['font'], 
                        1, 
                        (0, 0, 255), 
                        2
                    )
                    
                    # Check if collection is complete
                    if (elapsed_time >= self.time_window and feature_count >= self.min_frames):
                        try:
                            # Process collected features
                            features = []
                            while not self.shared_state.feature_queue.empty():
                                features.append(self.shared_state.feature_queue.get())
                            features = np.stack(features)
                            
                            # Preprocess and evaluate features
                            processed_features = self.preprocessor.process(features)
                            quality_info = self.quality_evaluator.evaluate_features(processed_features)
                            frame_display = self._draw_status_panel(frame_display, quality_info)
                            
                            if quality_info['total_score'] >= self.quality_threshold:
                                # Save features and update display
                                avg_feature = np.mean(processed_features, axis=0)
                                person_id = self._save_feature(avg_feature, quality_info)
                                
                                cv2.putText(
                                    frame_display,
                                    f"Success! ID: {person_id} Score: {quality_info['total_score']:.2f}",
                                    (10, h-30),
                                    self.display_config['font'], 
                                    1,
                                    self.display_config['colors']['success'], 
                                    2
                                )
                                
                                print(f"Registration complete! Person ID: {person_id}")
                                print(f"Quality score: {quality_info['total_score']:.2f}")
                                print("Metrics:", quality_info['metrics'])
                                
                            else:
                                # Display failure information
                                issues_text = " | ".join(quality_info['issues'])
                                cv2.putText(
                                    frame_display,
                                    f"Failed: {issues_text}",
                                    (10, h-30),
                                    self.display_config['font'], 
                                    1,
                                    self.display_config['colors']['warning'], 
                                    2
                                )
                                
                                print("Registration failed due to low quality")
                                print("Issues:", issues_text)
                                
                        except Exception as e:
                            print(f"Error processing features: {str(e)}")
                            cv2.putText(
                                frame_display,
                                "Error processing features",
                                (10, h-30),
                                self.display_config['font'], 
                                1,
                                self.display_config['colors']['warning'], 
                                2
                            )
                        
                        # Reset collection state
                        self.shared_state.is_collecting.value = False
                        self.shared_state.collecting_box[:] = [0.0, 0.0, 0.0, 0.0]
                        while not self.shared_state.feature_queue.empty():
                            try:
                                self.shared_state.feature_queue.get_nowait()
                            except:
                                pass
                
                # Display basic information
                cv2.putText(
                    frame_display,
                    f"Registered IDs: {len(self.shared_state.registered_ids)}", 
                    (10, 30),
                    self.display_config['font'], 
                    1, 
                    self.display_config['colors']['info'], 
                    2
                )
                
                # Display operation instructions
                cv2.putText(
                    frame_display,
                    "Click person to register | Press 'q' to quit",
                    (10, h-60),
                    self.display_config['font'], 
                    1.0,
                    self.display_config['colors']['text'], 
                    2
                )
                
                # Show frame
                cv2.imshow(self.display_name, frame_display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.control_event.set()
                    break
                
                frame_id += 1
                
            except queue.Empty:
                continue
                
            except cv2.error as e:
                print(f"OpenCV error: {str(e)}")
                window_created = False
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Consumer error: {str(e)}")
                time.sleep(0.1)
        
        cv2.destroyAllWindows()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_debug_window', action='store_true')
    parser.add_argument('--model', type=str, default='model/bytetrack_s.onnx')
    parser.add_argument('--reid_model', type=str, default='model/mgn_R50-ibn_dyn.onnx')
    parser.add_argument('--video', type=str, default='sample.mp4')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--score_th', type=float, default=0.1)
    parser.add_argument('--nms_th', type=float, default=0.7)
    parser.add_argument('--input_shape', type=str, default='608,1088')
    parser.add_argument('--reid_input_shape', type=str, default='384,128')
    parser.add_argument('--with_p6', action='store_true')
    parser.add_argument('--track_thresh', type=float, default=0.5)
    parser.add_argument('--track_buffer', type=int, default=30)
    parser.add_argument('--match_thresh', type=float, default=0.8)
    parser.add_argument('--min-box-area', type=float, default=10)
    parser.add_argument('--mot20', dest='mot20', default=False, action='store_true')
    
    args = parser.parse_args()
    return args

def main():
    # Set up multiprocessing
    mp.set_start_method('spawn')
    
    # Create interprocess communication objects
    frame_queue = mp.Queue(maxsize=30)
    control_event = mp.Event()
    shared_state = SharedState()
    
    # Create producer and consumer processes
    url = "rtsp://admin:zbzn2024@192.168.0.76:554"
    
    producer = VideoStreamProducer(url, frame_queue, control_event)
    
    consumer = PersonRegisterConsumer(
        frame_queue=frame_queue,
        control_event=control_event,
        shared_state=shared_state,
        save_dir='./person_features',
        display_name="Person Registration",
        time_window=10.0,
        min_frames=20,
        iou_threshold=0.5,
        quality_threshold=0.6
    )
    
    def cleanup_processes():
        """Helper function to clean up processes"""
        control_event.set()
        
        print("\nShutting down processes, please wait...")
        
        if consumer.is_alive():
            consumer.join(timeout=3)
            if consumer.is_alive():
                print("Force terminating consumer process")
                consumer.terminate()
                consumer.join(timeout=1)
        
        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except:
                pass
        
        if producer.is_alive():
            producer.join(timeout=3)
            if producer.is_alive():
                print("Force terminating producer process")
                producer.terminate()
                producer.join(timeout=1)
        
        if not consumer.is_alive():
            consumer.close()
        if not producer.is_alive():
            producer.close()
            
        print("Processes closed")
    
    try:
        producer.start()
        consumer.start()
        
        while True:
            try:
                if control_event.is_set() or not (producer.is_alive() and consumer.is_alive()):
                    break
                time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nReceived exit signal, closing...")
                break
                
    finally:
        cleanup_processes()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()