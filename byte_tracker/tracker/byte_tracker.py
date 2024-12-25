import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict

from .kalman_filter import KalmanFilter
from byte_tracker.tracker import matching
from .basetrack import BaseTrack, TrackState
from reid.reid_onnx import ReIDONNX

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score): 

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):#, person_id): # SJY Added person_id
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        # self.person_id = person_id # SJY Added person_id
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # self.reid_model = ReIDONNX(args)

    def update(self, output_results, img_info, img_size):
        """
        img_info: {'height': int, 'width': int, 'image': np.array}
        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        # img_h, img_w = img_info[0], img_info[1]
        img_h, img_w = img_info['height'], img_info['width']
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                        (tlbr, s) in zip(dets, scores_keep)]
            # print(img_info['image'].shape)

            # img = copy.deepcopy(img_info['image'])
            # for idet, tlbr in enumerate(dets):
            #     detection = detections[idet]
            #     tlbr = np.round(tlbr).astype(np.int32)
            #     tlbr = np.maximum(tlbr, 0)
            #     print("Frist", tlbr)
            #     box_img = img[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2]]
            #     print("Frist", box_img.shape)
            #     detection.curr_feature = self.reid_model.inference(box_img)[0]

        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # strack_pool = tracked_stracks
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        # TODO: 改成reid_feat
        # strack_pool_feat = ReIDFeatureExtractor.get_features(strack_pool, img, img0)
        # detections_feat = ReIDFeatureExtractor.get_features(detections, img, img0)
        # dists = matching.embedding_distance(strack_pool_feat, detections_feat)

        dists = matching.iou_distance(strack_pool, detections) 
        # dists = matching.embedding_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False) # 第一次匹配上的未追踪的轨迹，re_activate
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                        (tlbr, s) in zip(dets_second, scores_second)]
            
            # img = copy.deepcopy(img_info['image'])
            # for idet, tlbr in enumerate(dets_second):
            #     detection = detections_second[idet]
            #     tlbr = np.round(tlbr).astype(np.int32)
            #     tlbr = np.maximum(tlbr, 0)
            #     print("Second", tlbr)
            #     box_img = img[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2]]
            #     print("Second", box_img.shape)
            #     detection.curr_feature = self.reid_model.inference(box_img)[0]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        # TODO: 记录稳定轨迹
        # img = copy.deepcopy(img_info['image'])
        # stable_tracks = []
        # for track in output_stracks:
        #     tlbr = track.tlbr
        #     tlbr = np.round(tlbr).astype(np.int32)
        #     box_img = img[tlbr [1]:tlbr[3], tlbr[0]:tlbr[2]]

        #     if track.tracklet_len > 30 and box_img.shape[0] > 0 and box_img.shape[1] > 0: # 如果追踪长度超过10,则比对ReID特征
        #         # tlbr = STrack.tlwh_to_tlbr(track._tlwh)
        #         # tlbr = track.tlbr
        #         # tlbr = np.round(tlbr).astype(np.int32)
        #         # box_img = img[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2]]
        #         print(track.track_id, box_img.shape, track)
        #         track.curr_feature = self.reid_model.inference(box_img)[0]
        #         stable_tracks.append(track)
            # print(f"track_id: {track.track_id}, tracklet_len: {track.tracklet_len}")

        return output_stracks


def joint_stracks(tlista: List[STrack], tlistb: List[STrack]) -> List[STrack]:
    """
    合并两个轨迹列表，确保不存在重复的track_id
    Args:
        tlista: 第一个轨迹列表
        tlistb: 第二个轨迹列表
    Returns:
        合并后的轨迹列表
    """
    exists = {}
    res = []

    # 添加第一个列表中的所有轨迹
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)

    # 添加第二个列表中不重复的轨迹
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)

    return res

def sub_stracks(tlista: List[STrack], tlistb: List[STrack]) -> List[STrack]:
    """
    从列表A中删除在列表B中出现的轨迹
    Args:
        tlista: 被减列表
        tlistb: 要减去的列表
    Returns:
        差集列表
    """
    # stracks = {}
    # for t in tlista:
    #     stracks[t.track_id] = t
    # for t in tlistb:
    #     tid = t.track_id
    #     if stracks.get(tid, 0):
    #         del stracks[tid]
    # return list(stracks.values())

    track_ids_b = {t.track_id for t in tlistb}
    return [t for t in tlista if t.track_id not in track_ids_b]


def remove_duplicate_stracks(stracksa: List[STrack], stracksb: List[STrack]) -> Tuple[List[STrack], List[STrack]]:
    """
    移除重复的轨迹，保留跟踪时间较长的轨迹
    Args:
        stracksa: 第一个轨迹列表
        stracksb: 第二个轨迹列表
    Returns:
        处理后的两个轨迹列表，确保没有重复轨迹
    """
    # 计算两个列表中轨迹之间的IOU距离
    pdist = matching.iou_distance(stracksa, stracksb)
    # 找出IOU重叠度高的轨迹对(阈值0.15)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    
    # 对每对重叠的轨迹，保留跟踪时间更长的轨迹
    for p, q in zip(*pairs):
        # 计算两个轨迹的跟踪时长
        timep = stracksa[p].frame_id - stracksa[p].start_frame  # 第一个轨迹的存活时间
        timeq = stracksb[q].frame_id - stracksb[q].start_frame  # 第二个轨迹的存活时间
        
        # 比较存活时间，将存活时间短的轨迹标记为重复
        if timep > timeq:  # 如果第一个轨迹存活时间更长
            dupb.append(q)  # 标记第二个轨迹为重复
        else:  # 如果第二个轨迹存活时间更长或相等
            dupa.append(p)  # 标记第一个轨迹为重复
            
    # 从两个列表中分别移除被标记为重复的轨迹
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]  # 保留未标记为重复的轨迹
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]  
    
    return resa, resb  # 返回处理后的两个轨迹列表

# 加载本地注册过的特征向量
def load_registered_features(feats_dir: str):
    import glob
    feats = []
    feats_ids = []

    feat_files = glob.glob(os.path.join(feats_dir, '*.npz'))
    for feat_path in feat_files:
        person_id = int(os.path.basename(feat_path).split('_')[-1].split('.')[0])
        feat = np.load(feat_path)['feature']
        feat = feat / np.linalg.norm(feat, keepdims=True)

        feats.append(feat)
        feats_ids.append(person_id)

    return feats, feats_ids
