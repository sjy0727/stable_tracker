import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

# from cython_bbox import bbox_overlaps as bbox_ious
from byte_tracker.tracker import kalman_filter
import time

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    """
    线性分配算法 - 使用Jonker-Volgenant算法求解带阈值的线性分配问题
    
    参数:
        cost_matrix: 成本矩阵，表示所有可能分配的代价
        thresh: 阈值，超过此代价的分配将被忽略
    
    返回:
        matches: 匹配的行列索引对数组
        unmatched_a: 未匹配的行索引数组
        unmatched_b: 未匹配的列索引数组
    """
    # 如果成本矩阵为空，返回空结果
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    
    # 初始化结果列表
    matches, unmatched_a, unmatched_b = [], [], []
    
    # 使用lap.lapjv求解线性分配问题
    # extend_cost=True表示扩展成本矩阵以处理不完全匹配
    # cost_limit=thresh设置代价上限
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    
    # 遍历行索引的匹配结果
    for ix, mx in enumerate(x):
        # 如果存在有效匹配（mx >= 0），添加到matches列表
        if mx >= 0:
            matches.append([ix, mx])
    
    # 获取未匹配的行索引（x < 0的位置）
    unmatched_a = np.where(x < 0)[0]
    # 获取未匹配的列索引（y < 0的位置）
    unmatched_b = np.where(y < 0)[0]
    
    # 将matches转换为numpy数组
    matches = np.asarray(matches)
    
    return matches, unmatched_a, unmatched_b

def bbox_ious_vectorized(boxes, query_boxes):
    """
    使用NumPy向量化操作计算两组边界框之间的IoU
    
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    
    # 将boxes扩展为(N,1,4)
    boxes = boxes.reshape(N,1,4)
    # 将query_boxes扩展为(1,K,4)
    query_boxes = query_boxes.reshape(1,K,4)
    
    # 计算所有boxes的面积 (N,1)
    boxes_area = (boxes[:, :, 2] - boxes[:, :, 0] + 1) * \
                (boxes[:, :, 3] - boxes[:, :, 1] + 1)
    
    # 计算所有query_boxes的面积 (1,K)
    query_boxes_area = (query_boxes[:, :, 2] - query_boxes[:, :, 0] + 1) * \
                    (query_boxes[:, :, 3] - query_boxes[:, :, 1] + 1)
    
    # 计算交集区域的左上角和右下角坐标
    iw = np.minimum(boxes[:, :, 2], query_boxes[:, :, 2]) - \
        np.maximum(boxes[:, :, 0], query_boxes[:, :, 0]) + 1
    ih = np.minimum(boxes[:, :, 3], query_boxes[:, :, 3]) - \
        np.maximum(boxes[:, :, 1], query_boxes[:, :, 1]) + 1
    
    # 将无效值（负值）设为0
    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)
    
    # 计算交集面积
    intersection = iw * ih
    
    # 计算并集面积 (N,K)
    union = boxes_area + query_boxes_area - intersection
    
    # 计算IoU
    overlaps = intersection / union
    
    return overlaps

def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if ious.size == 0:
        return ious

    ious = bbox_ious_vectorized(
        np.ascontiguousarray(atlbrs, dtype=np.float32),
        np.ascontiguousarray(btlbrs, dtype=np.float32)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def feature_distance(tracks, feats, metric='cosine'):
    cost_matrix = np.zeros((len(tracks), len(feats)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    features = np.asarray(feats, dtype=np.float32)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    
    # TODO:smooth_feat 没有定义
    track_features = np.asarray([track.curr_feature for track in tracks], dtype=np.float32)
    # track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    print(track_features.shape, features.shape)
    cost_matrix = np.maximum(0.0, cdist(track_features, features, metric))  # Nomalized features
    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feature for track in detections], dtype=np.float32)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    
    # TODO:smooth_feat 没有定义
    track_features = np.asarray([track.curr_feature for track in tracks], dtype=np.float32)
    # track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost