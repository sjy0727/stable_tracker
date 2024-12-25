#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy

import numpy as np
import onnxruntime

from reid.reid_onnx import ReIDONNX

from byte_tracker.utils.yolox_utils import (
    pre_process,
    post_process,
    multiclass_nms,
)
from byte_tracker.tracker.byte_tracker import BYTETracker, load_registered_features
from byte_tracker.tracker.matching import feature_distance, linear_assignment
from byte_tracker.tracker.stable_cascade_match import cascade_reid_matching


class ByteTrackerONNX(object):
    def __init__(self, args):
        self.args = args

        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        self.session = onnxruntime.InferenceSession(args.model, providers=onnxruntime.get_available_providers())
        self.input_shape = tuple(map(int, args.input_shape.split(',')))

        self.tracker = BYTETracker(args, frame_rate=30)
        self.reid_model = ReIDONNX(args)
        self.local_features, self.local_features_ids  = load_registered_features('person_features')
        print("self.local_features_ids", self.local_features_ids)

        self.matcher = EnhancedMatcher(
            threshold=0.5,
            position_threshold=100,
            time_window=30,
            decay_period=30
        )

    def _pre_process(self, image):
        image_info = {'id': 0}

        image_info['image'] = copy.deepcopy(image)
        image_info['width'] = image.shape[1]
        image_info['height'] = image.shape[0]

        preprocessed_image, ratio = pre_process(
            image,
            self.input_shape,
            self.rgb_means,
            self.std,
        ) # BGR->RGB
        image_info['ratio'] = ratio

        return preprocessed_image, image_info

    def inference(self, image):
        image, image_info = self._pre_process(image) 

        input_name = self.session.get_inputs()[0].name

        result = self.session.run(None, {input_name: image[None, :, :, :]})
        # print('result[0].shape: ', result[0].shape)
        dets = self._post_process(result, image_info) # NMS + xywh2xyxy

        bboxes, ids, scores, stable_tracks = self._tracker_update(
            dets,
            image_info,
        )

        # TODO: 1. 已注册人全在屏幕中, 
        # TODO: 2. 已注册人部分在屏幕中
        # TODO: 3. 已注册人全部不在屏幕中
        # cost_matrix = feature_distance(stable_tracks, self.local_features)
        # # print(cost_matrix)
        # matches, u_tracks, u_feats = linear_assignment(1 - cost_matrix, 0.5)
        # print(cost_matrix.argmax(1))
        # for itrack, ifeat in matches:
        #     print("itrack, ifeat", itrack, ifeat)
        #     track = stable_tracks[itrack]
        #     track.track_id = self.local_features_ids[ifeat]
        # for it in u_tracks:
        #     track = stable_tracks[it]
        #     track.track_id = 999

        ######
        if stable_tracks:
            track_feats = np.array([track.curr_feature for track in stable_tracks])
            track_positions = np.array([track.tlwh for track in stable_tracks])
            track_ids = np.array([track.track_id for track in stable_tracks])
            # print("track_feats",track_feats)
            # print("track_positions",track_positions)
            # print("track_ids",track_ids)
            
            # matches = self.matcher.match_with_constraints(
            #     track_feats=track_feats,
            #     track_positions=track_positions,
            #     track_ids=track_ids,
            #     reg_feats=np.array(self.local_features),
            #     reg_ids=np.array(self.local_features_ids)
            # )

            matches = cascade_reid_matching(track_feats, np.array(self.local_features), 0.1)
            print("matches", matches)
            for track_idx, reg_idx, score in matches:
                print(reg_idx)
                # stable_tracks[track_idx].track_id = self.local_features_ids[reg_idx]
                stable_tracks[track_idx].track_id = self.local_features_ids[reg_idx]



        return image_info, bboxes, ids, scores

    def _post_process(self, result, image_info):
        predictions = post_process(
            result[0],
            self.input_shape,
            p6=self.args.with_p6,
        )
        predictions = predictions[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        # boxes_xyxy = boxes
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= image_info['ratio']

        dets = multiclass_nms(
            boxes_xyxy,
            scores,
            nms_thr=self.args.nms_th,
            score_thr=self.args.score_th,
        )

        return dets

    def _tracker_update(self, dets, image_info):
        online_targets = []
        stable_tracks = []
        if dets is not None:
            # dets xywh conf cls
            # print('dets shape', dets[:, :-1].shape, dets[:, -1])
            online_targets = self.tracker.update(
                dets[:, :-1], # [x1, y1, x2, y2, score, cls] -> [x1, y1, x2, y2, score]
                # [image_info['height'], image_info['width']],
                image_info,
                [image_info['height'], image_info['width']],
            )

        online_tlwhs = []
        online_ids = []
        online_scores = []
        img = copy.deepcopy(image_info['image'])
        for online_target in online_targets:
            tlwh = online_target.tlwh
            track_id = online_target.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(track_id)
                online_scores.append(online_target.score)

                # TODO: 记录稳定轨迹
                _tlwh = np.round(copy.deepcopy(tlwh)).astype(np.int32)
                _tlwh = np.maximum(_tlwh, 0)
                if online_target.tracklet_len > 30 and _tlwh[2] > 0 and _tlwh[3] > 0:
                    box_img = img[_tlwh[1]:_tlwh[1] + _tlwh[3], _tlwh[0]:_tlwh[0] + _tlwh[2]]
                    print(online_target)
                    online_target.curr_feature = self.reid_model.inference(box_img)[0]
                    # print(online_target.curr_feature.shape)
                    stable_tracks.append(online_target)

        return online_tlwhs, online_ids, online_scores, stable_tracks

class EnhancedMatcher:
    def __init__(self, threshold=0.75, position_threshold=100, time_window=30, decay_period=30):
        self.threshold = threshold  # ReID特征匹配阈值
        self.position_threshold = position_threshold  # 位置距离阈值(像素)
        self.time_window = time_window  # 时序窗口大小(帧)
        self.decay_period = decay_period  # 时间衰减周期
        self.historical_matches = {}  # 历史匹配记录 {track_id: [records]} # FIXME:OOM
        self.frame_id = 0

    def match_with_constraints(self, 
                            track_feats,      # 跟踪目标的ReID特征 (M, 2048) 
                            track_positions,   # 跟踪目标的位置 (M, 4) [x,y,w,h]
                            track_ids,        # 跟踪目标的ID列表 (M,)
                            reg_feats,        # 注册人员特征 (N, 2048)
                            reg_ids):         # 注册人员ID列表 (N,)
        """结合多种约束的特征匹配"""
        self.frame_id += 1
        matches = []  # [(track_idx, reg_idx, score)]

        # 清理过期的历史记录
        for track_id in self.historical_matches:
            self.historical_matches[track_id] = [
                record for record in self.historical_matches[track_id]
                if self.frame_id - record['frame_id'] <= self.time_window
            ]

        # 1. 计算ReID特征相似度矩阵
        # similarity_matrix = self._compute_similarity(track_feats, reg_feats)
        from scipy.spatial.distance import cdist
        similarity_matrix = cdist(track_feats, reg_feats, 'cosine')


        # 2. 遍历每个跟踪目标
        for track_idx, track_id in enumerate(track_ids):
            track_pos = track_positions[track_idx] 
            track_history = self.historical_matches.get(track_id, [])
            
            # 计算所有可能的匹配分数
            match_scores = []
            for reg_idx, reg_id in enumerate(reg_ids):
                reid_sim = similarity_matrix[track_idx, reg_idx]
                
                # 各种约束条件的分数
                temporal_score = self._temporal_score(track_history, reg_id)
                position_score = self._position_score(track_pos, track_history, reg_id)
                
                # 综合评分
                final_score = self._combine_scores(
                    reid_sim=reid_sim,
                    temporal_score=temporal_score,
                    position_score=position_score
                )
                
                match_scores.append((reg_idx, final_score))
            
            # 选择最佳匹配
            if match_scores:
                best_reg_idx, best_score = max(match_scores, key=lambda x: x[1])
                reid_sim = similarity_matrix[track_idx, best_reg_idx]
                print("reid_sim", reid_sim)
                
                if best_score > self.threshold and reid_sim > 0.5:  # 确保ReID相似度也要足够高
                    matches.append((track_idx, best_reg_idx, best_score))
                    
                    # 更新历史记录
                    if track_id not in self.historical_matches:
                        self.historical_matches[track_id] = []

                    self.historical_matches[track_id].append({
                        'frame_id': self.frame_id,
                        'reg_id': reg_ids[best_reg_idx],
                        'score': best_score,
                        'position': track_positions[track_idx]
                    })
        # print("matches", matches)
        # print("self.historical_matches", self.historical_matches)
        return matches

    def _temporal_score(self, track_history, reg_id):
        """
        计算时序一致性分数
        Args:
            track_history: List[{
                'frame_id': int,      # 帧ID
                'reg_id': int,        # 匹配的注册ID 
                'score': float,       # 匹配分数
                'position': ndarray   # 位置信息[x,y,w,h]
            }] 轨迹的历史匹配记录
            reg_id: int 待匹配的注册ID
        Returns:
            temporal_score: float 时序一致性分数 [0,1]
        """
        if not track_history:
            return 0.0
                
        # 只考虑时间窗口内的历史记录
        recent_history = [
            record for record in track_history 
            if self.frame_id - record['frame_id'] <= self.time_window
        ]
        
        if not recent_history:
            return 0.0
        
        # 计算该reg_id在最近记录中的匹配情况
        matches_with_id = [
            record for record in recent_history 
            if record['reg_id'] == reg_id
        ]
        
        # 匹配频率
        match_freq = len(matches_with_id) / len(recent_history)
        
        # 考虑最近匹配的时间间隔
        if matches_with_id:
            last_match_frame = matches_with_id[-1]['frame_id']
            frames_gap = self.frame_id - last_match_frame
            # 时间衰减系数: 间隔越大,系数越小
            time_decay = np.exp(-frames_gap / self.decay_period)
        else:
            time_decay = 0.0
        
        # 最终分数结合匹配频率和时间衰减
        temporal_score = 0.7 * match_freq + 0.3 * time_decay
        
        return temporal_score

    def _position_score(self, current_pos, track_history, reg_id):
        """
        计算空间位置约束分数
        """
        if not track_history:
            return 1.0
            
        # 获取时间窗口内最近一次同ID的匹配记录
        recent_matches = [
            record for record in track_history 
            if record['reg_id'] == reg_id and 
            self.frame_id - record['frame_id'] <= self.time_window
        ]
        
        if not recent_matches:
            return 1.0
            
        last_match = recent_matches[-1]
        frames_gap = self.frame_id - last_match['frame_id']
        
        # 根据帧数差估算合理的运动范围
        reasonable_move_range = frames_gap * 50
        
        # 计算当前和历史位置的中心点
        current_center = np.array([
            current_pos[0] + current_pos[2]/2, 
            current_pos[1] + current_pos[3]/2
        ])
        last_pos = last_match['position']
        last_center = np.array([
            last_pos[0] + last_pos[2]/2, 
            last_pos[1] + last_pos[3]/2
        ])
        
        # 计算位置偏移
        distance = np.linalg.norm(current_center - last_center)
        
        # 计算分数
        if distance <= reasonable_move_range:
            position_score = 1.0
        else:
            # 采用平滑的指数衰减,而不是线性衰减
            position_score = np.exp(-(distance - reasonable_move_range) / self.position_threshold)
            
        return position_score

    def _compute_similarity(self, track_feats, reg_feats):
        """计算ReID特征相似度"""
        # 添加数值稳定性处理
        track_feats = track_feats.astype(np.float32)
        reg_feats = reg_feats.astype(np.float32)
        
        similarity = np.dot(track_feats, reg_feats.T)
        track_norms = np.linalg.norm(track_feats, axis=1, keepdims=True)
        reg_norms = np.linalg.norm(reg_feats, axis=1, keepdims=True)
        
        # 避免除零
        track_norms = np.maximum(track_norms, 1e-6)
        reg_norms = np.maximum(reg_norms, 1e-6)
        
        return similarity / (track_norms @ reg_norms.T)

    def _combine_scores(self, reid_sim, temporal_score, position_score):
        """组合多个分数"""
        weights = {
            'reid': 0.6,      # ReID特征权重
            'temporal': 0.3,  # 时序一致性权重
            'position': 0.1   # 位置约束权重
        }
        
        return (weights['reid'] * reid_sim +
                weights['temporal'] * temporal_score + 
                weights['position'] * position_score)

def test_matcher():
    """测试用例"""
    # 初始化matcher
    matcher = EnhancedMatcher(
        threshold=0.75,
        position_threshold=100,
        time_window=30,
        decay_period=30
    )
    
    # 模拟数据
    M, N = 5, 10  # M个跟踪目标, N个注册人员
    track_feats = np.random.randn(M, 2048)
    track_positions = np.random.rand(M, 4) * 1000  # 随机位置
    track_ids = np.arange(M)
    reg_feats = np.random.randn(N, 2048)
    reg_ids = np.arange(N)
    
    # 特征归一化
    track_feats = track_feats / np.linalg.norm(track_feats, axis=1, keepdims=True)
    reg_feats = reg_feats / np.linalg.norm(reg_feats, axis=1, keepdims=True)
    
    # 执行匹配
    matches = matcher.match_with_constraints(
        track_feats,
        track_positions,
        track_ids,
        reg_feats,
        reg_ids
    )
    
    return matches