import copy
import argparse

import cv2
import onnxruntime
import numpy as np

def get_test_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--use_debug_window',
        action='store_true',
    )

    parser.add_argument(
        '--model',
        type=str,
        default='byte_tracker/model/bytetrack_s.onnx',
    )

    parser.add_argument(
        '--reid_model',
        type=str,
        default='./model/mgn_R50-ibn_dyn.onnx',
    )

    parser.add_argument(
        '--video',
        type=str,
        default='sample.mp4',
        # default='twitter_x264.mp4',
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
        default='384,128',  # 改为字符串格式
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
        default=0.5,
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

class ReIDONNX(object):
    def __init__(self, args):
        self.args = args
        self.session = onnxruntime.InferenceSession(args.reid_model, providers=onnxruntime.get_available_providers())
        self.reid_input_shape = tuple(map(int, args.reid_input_shape.split(',')))

    def _pre_process(self, image, after_det=True):
        image_info = {'id': 0}
        
        # 确保图像是BGR格式且维度为(H,W,C)
        if image.shape[0] == 3:  # 如果是(C,H,W)格式
            image = np.transpose(image, (1, 2, 0))
        
        image_info['image'] = copy.deepcopy(image)
        image_info['height'] = self.reid_input_shape[0]
        image_info['width'] = self.reid_input_shape[1]
        
        # 调整图像大小
        preprocessed_image = cv2.resize(image, 
                                    dsize=(self.reid_input_shape[1], self.reid_input_shape[0]), 
                                    interpolation=cv2.INTER_CUBIC)
        
        # 标准化处理
        preprocessed_image = preprocessed_image.astype(np.float32) / 255.0
        
        # 转换为模型需要的格式(C,H,W)
        preprocessed_image = np.transpose(preprocessed_image, (2, 0, 1))
        preprocessed_image = preprocessed_image[::-1, ...] # 将BGR转换为RGB
        # print("preprocessed_image",preprocessed_image.shape)
        
        return preprocessed_image, image_info
    
    def _normalize(self, nparray, order=2, axis=-1):
        norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
        return nparray / (norm + np.finfo(np.float32).eps)
    
    def inference(self, image):
        image, image_info = self._pre_process(image)
        
        input_name = self.session.get_inputs()[0].name
        # 转换为模型需要的格式(N,C,H,W)
        result = self.session.run(None, {input_name: image[None, :, :, :]})[0]

        reid_feat = self._normalize(result)  # 取第一个输出并进行归一化
        
        return reid_feat
    
    def multi_inference(self, images):
        pass

if __name__ == '__main__':
    args = get_test_args()

    reid = ReIDONNX(args)
    
    # 创建一个随机测试图像 (H,W,C)格式
    dummy_img = np.random.rand(640, 640, 3).astype(np.float32)
    feat = reid.inference(dummy_img)
    print(feat.shape)