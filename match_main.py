import cv2
import numpy as np
import multiprocessing as mp
import time
import signal
import copy
import sys
import queue
from typing import Optional
from multiprocessing.synchronize import Event
from loguru import logger
from byte_tracker.tracker.byte_tracker_onnx import ByteTrackerONNX

# from ultralytics import YOLO
from reid.reid_onnx import ReIDONNX
from main import get_args

class VideoStreamProducer(mp.Process):
    def __init__(self, url: str, 
                frame_queue: mp.Queue, 
                control_event: Event,
                max_queue_size: int = 30):
        """
        初始化视频流生产者进程
        
        Args:
            url: 视频流地址
            frame_queue: 进程间共享的帧队列
            control_event: 用于控制进程的事件
            max_queue_size: 队列最大容量
        """
        super().__init__()
        self.url = url
        self.frame_queue = frame_queue
        self.control_event = control_event
        self.max_queue_size = max_queue_size
        
    def run(self):
        """生产者进程的主函数"""
        # 设置信号处理
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        
        cap: Optional[cv2.VideoCapture] = None
        retry_count = 0
        max_retries = 3
        
        while not self.control_event.is_set():
            try:
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(self.url)
                    if not cap.isOpened():
                        raise ConnectionError(f"无法连接到视频流: {self.url}")
                
                ret, frame = cap.read()
                if not ret or frame is None:
                    raise ConnectionError("读取视频帧失败")
                    
                # 检查帧格式
                if len(frame.shape) != 3:
                    raise ValueError("无效的帧格式")
                
                # 将帧转换为可通过队列传输的格式 
                # TODO: 定义DTO
                frame_data = {
                    'frame': frame,
                    'timestamp': time.time()
                }
                
                # 非阻塞方式放入队列
                if not self.frame_queue.full():
                    self.frame_queue.put(frame_data, block=False)
                retry_count = 0
                
            except (queue.Full, BrokenPipeError):
                # 队列满或管道断开,等待一下
                time.sleep(0.01)
                continue
                
            except Exception as e:
                print(f"生产者错误: {str(e)}")
                retry_count += 1
                
                if retry_count >= max_retries:
                    print("达到最大重试次数,退出生产者进程")
                    break
                    
                if cap is not None:
                    cap.release()
                    cap = None
                    
                time.sleep(1)
        
        # 清理资源
        if cap is not None:
            cap.release()

class VideoStreamConsumer(mp.Process):
    def __init__(self, frame_queue: mp.Queue, 
                control_event: Event,
                process_frame_call_back: callable,
                display_name: str = "Video Stream",
                show: bool = True):
        """
        初始化视频流消费者进程
        
        Args:
            frame_queue: 进程间共享的帧队列
            control_event: 用于控制进程的事件
            display_name: 显示窗口名称
        """
        super().__init__()
        self.frame_queue = frame_queue
        self.control_event = control_event
        self.process_frame_call_back = process_frame_call_back
        self.display_name = display_name
        self.show = show
        
    def run(self):
        """消费者进程的主函数"""
        # 设置信号处理
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        
        window_created = False

        # FIXME: 把初始化onnx模型的操作放到子进程中，且不要在__init__中初始化，否则报错 https://blog.csdn.net/qq_36276587/article/details/119913628
        # TypeError: cannot pickle 'onnxruntime.capi.onnxruntime_pybind11_state.InferenceSession' object
        # 模型初始化
        # det_model = YOLO("./model/yolov8s.pt")
        # reid_model = ReIDONNX(args=get_args())
        byte_tracker = ByteTrackerONNX(args=get_args())
        
        frame_id = 1
        
        while not self.control_event.is_set():
            try:
                # 非阻塞方式获取帧
                if self.frame_queue.empty():
                    time.sleep(0.01)
                    continue
                
                start_time = time.time()
                # TODO: 取出帧数据
                frame_data = self.frame_queue.get(block=False)
                frame = frame_data['frame']
                
                debug_image = copy.deepcopy(frame)

                image_info, bboxes, ids, scores = byte_tracker.inference(frame)

                elapsed_time = time.time() - start_time
                

                # 处理帧
                if self.process_frame_call_back:
                    try:
                        processed_frame = self.process_frame_call_back(
                            debug_image,
                            bboxes,
                            ids,
                            scores,
                            frame_id,
                            elapsed_time,)
                        
                        if processed_frame is None:
                            continue

                    except Exception as e:
                        print(f"帧处理回调错误: {str(e)}")
                        continue
                
                # 创建或重新创建窗口
                if not window_created:
                    cv2.namedWindow(self.display_name, cv2.WINDOW_NORMAL)
                    window_created = True
                
                if self.show:
                    # 显示帧
                    cv2.imshow(self.display_name, processed_frame)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        self.control_event.set()
                        break
                
                logger.info('frame {}/{} ({:.2f} ms)'.format(frame_id, -1, elapsed_time * 1000), )

                frame_id += 1

            except queue.Empty:
                continue
                
            except cv2.error as e:
                print(f"OpenCV错误: {str(e)}")
                # 尝试重新创建窗口
                window_created = False
                time.sleep(0.1)
                
            except Exception as e:
                print(f"消费者错误: {str(e)}")
                time.sleep(0.1)
        
        # 清理资源
        cv2.destroyAllWindows()

def process_frame(frame: np.ndarray) -> Optional[np.ndarray]:
        """
        处理视频帧的方法,可以在子类中重写以添加自定义处理逻辑
        
        Args:
            frame: 输入帧
            
        Returns:
            处理后的帧,如果处理失败返回None
        """
        try:
            # 这里可以添加自定义的帧处理逻辑
            # 例如添加时间戳
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return frame
        except Exception as e:
            print(f"帧处理错误: {str(e)}")
            return None

def mouse_callback(event, x, y, flags, param):
    pass

def get_id_color(index):
    temp_index = abs(int(index)) * 3
    color = ((37 * temp_index) % 255, (17 * temp_index) % 255,
             (29 * temp_index) % 255)
    return color

def draw_tracking_info(
    image,
    tlwhs,
    ids,
    scores,
    frame_id=0,
    elapsed_time=0.,
):
    text_scale = 1.5
    text_thickness = 2
    line_thickness = 2

    text = 'frame: %d ' % (frame_id)
    text += 'elapsed time: %.0fms ' % (elapsed_time * 1000)
    text += 'num: %d' % (len(tlwhs))
    cv2.putText(
        image,
        text,
        (0, int(15 * text_scale)),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (0, 255, 0),
        thickness=text_thickness,
    )

    for index, tlwh in enumerate(tlwhs):
        x1, y1 = int(tlwh[0]), int(tlwh[1])
        x2, y2 = x1 + int(tlwh[2]), y1 + int(tlwh[3])

        color = get_id_color(ids[index])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

        text = str(ids[index]) + ':%.2f' % (scores[index])
        # text = str(ids[index])
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN,
                    text_scale, (0, 0, 0), text_thickness + 3)
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN,
                    text_scale, (255, 255, 255), text_thickness)
    return image

def main():
    # 设置进程启动方法
    mp.set_start_method('spawn')
    
    # 创建进程间通信对象
    frame_queue = mp.Queue(maxsize=30)
    control_event = mp.Event()
    
    # 创建生产者和消费者进程
    url = "rtsp://admin:zbzn2024@192.168.0.76:554"  # 实际的视频流地址

    producer = VideoStreamProducer(url, frame_queue, control_event)
    consumer = VideoStreamConsumer(frame_queue, control_event, draw_tracking_info)
    
    def cleanup_processes():
        """清理进程的辅助函数"""
        # 设置控制事件通知进程退出
        control_event.set()
        
        print("\n正在关闭进程，请稍候...")
        
        # 等待消费者进程结束
        if consumer.is_alive():
            consumer.join(timeout=3)
            if consumer.is_alive():
                print("强制终止消费者进程")
                consumer.terminate()
                consumer.join(timeout=1)
        
        # 清空队列
        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except:
                pass
        
        # 等待生产者进程结束
        if producer.is_alive():
            producer.join(timeout=3)
            if producer.is_alive():
                print("强制终止生产者进程")
                producer.terminate()
                producer.join(timeout=1)
        
        # 安全关闭进程
        if not consumer.is_alive():
            consumer.close()
        if not producer.is_alive():
            producer.close()
            
        print("进程已关闭")

    try:
        # 启动进程
        producer.start()
        consumer.start()
        
        # 等待进程结束或用户中断
        while True:
            try:
                if control_event.is_set() or not (producer.is_alive() and consumer.is_alive()):
                    break
                time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n收到退出信号，正在关闭...")
                break
                
    finally:
        cleanup_processes()

if __name__ == "__main__":
    main()