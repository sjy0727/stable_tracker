import numpy as np
from scipy.spatial.distance import cdist

def cascade_reid_matching(query_features, gallery_features, threshold=0.5):
    """
    执行级联式ReID特征匹配，处理查询特征与底库特征之间的匹配关系
    
    参数说明:
        query_features: 形状为(M, 2048)的numpy数组，M为检测到的行人数量
        gallery_features: 形状为(N, 2048)的numpy数组，N为注册人员数量
        threshold: 有效匹配的最小相似度阈值
        
    返回值:
        matches: 包含(查询索引, 底库索引, 相似度)的元组列表，表示有效的匹配结果
    """
    # 特征向量归一化，确保计算余弦相似度
    # query_features = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
    # gallery_features = gallery_features / np.linalg.norm(gallery_features, axis=1, keepdims=True)
    
    # # 使用矩阵乘法计算相似度矩阵（余弦相似度）
    # similarity_matrix = np.dot(query_features, gallery_features.T)  # 形状: (M, N)

    similarity_matrix = np.maximum(0.0, 1 - cdist(query_features, gallery_features, metric='cosine'))
    print("similarity_matrix", similarity_matrix)
    # 初始化级联匹配所需的变量
    matches = []  # 存储最终的匹配结果
    used_query_indices = set()  # 已匹配的查询索引集合
    used_gallery_indices = set()  # 已匹配的底库索引集合
    
    while True:
        # 创建未使用索引的掩码
        query_mask = ~np.isin(np.arange(similarity_matrix.shape[0]), list(used_query_indices))
        gallery_mask = ~np.isin(np.arange(similarity_matrix.shape[1]), list(used_gallery_indices))
        
        # 使用掩码获取有效的相似度子矩阵
        valid_similarities = similarity_matrix[query_mask][:, gallery_mask]
        
        # 如果没有剩余的有效匹配对，退出循环
        if valid_similarities.size == 0:
            break
        
        # 找出最高的相似度值
        max_sim = np.max(valid_similarities)
        # 如果最高相似度低于阈值，退出循环
        if max_sim < threshold:
            break
            
        # 获取最佳匹配对的局部索引
        query_idx_local = np.argmax(valid_similarities) // valid_similarities.shape[1]
        gallery_idx_local = np.argmax(valid_similarities) % valid_similarities.shape[1]
        
        # 将局部索引转换为全局索引
        query_idx = np.arange(similarity_matrix.shape[0])[query_mask][query_idx_local]
        gallery_idx = np.arange(similarity_matrix.shape[1])[gallery_mask][gallery_idx_local]
        
        # 将匹配结果添加到列表中
        matches.append((query_idx, gallery_idx, max_sim))
        
        # 更新已使用的索引集合
        used_query_indices.add(query_idx)
        used_gallery_indices.add(gallery_idx)
    
    return matches

def get_reid_results(matches, query_features, gallery_features):
    """
    将匹配结果转换为更详细的结果格式
    
    参数说明:
        matches: 包含(查询索引, 底库索引, 相似度)的匹配结果列表
        query_features: 原始查询特征数组
        gallery_features: 原始底库特征数组
        
    返回值:
        results: 包含匹配详细信息的字典
    """
    results = {
        'matched_pairs': matches,  # 匹配对列表
        'num_matches': len(matches),  # 匹配数量
        'query_coverage': len(set(m[0] for m in matches)) / len(query_features),  # 查询集覆盖率
        'gallery_coverage': len(set(m[1] for m in matches)) / len(gallery_features),  # 底库覆盖率
        'average_similarity': np.mean([m[2] for m in matches]) if matches else 0  # 平均相似度
    }
    return results

# 使用示例
def demo_matching():
    """
    演示ReID匹配功能的示例函数
    """
    # 生成示例数据
    M, N = 3, 2  # 5个查询样本，3个底库样本
    query_features = np.random.randn(M, 2048)  # 随机生成查询特征
    # gallery_features = np.random.randn(N, 2048)  # 随机生成底库特征
    gallery_features = query_features[:3]  # 随机生成底库特征

    # 执行匹配
    matches = cascade_reid_matching(query_features, gallery_features, threshold=0.5)
    results = get_reid_results(matches, query_features, gallery_features)
    
    return results

# 使用方法示例：
"""
# 准备特征向量数据
query_features = ...  # 形状: (M, 2048)
gallery_features = ... # 形状: (N, 2048)

# 执行级联匹配
matches = cascade_reid_matching(query_features, gallery_features, threshold=0.5)

# 获取详细的匹配结果
results = get_reid_results(matches, query_features, gallery_features)
"""
if __name__ == '__main__':
    print(demo_matching())