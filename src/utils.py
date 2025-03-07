import numpy as np

# 모든 데이터와 중심점의 유클리드 거리 계산
def euclidean_distance(data, centroids):
    return np.sqrt(np.sum((data-centroids)**2, axis=1))




def compute_distance(data, centroids):
    """
    해당 데이터 포인트와 모든 centroid 사이의 거리를 계산
    """
    
    n_samples=data.shape[0] # data 행 개수
    k=centroids.shape[0]    # centroids 행개수
    
    # 거리 행렬 초기화 distance는 데이터 포인트와 모든 중심점 사이의 거리를 저장할 행렬
    distances = np.zeros((n_samples, k)) # n_samples는 data 행 개수, k는 centroids 행 개수(=cluster 개수)
    
    # 모든 데이터 행들과 centroid들의 유클리드 거리를 구함.
    for i in range(k):
        distances[:,i] = euclidean_distance(data,centroids[i]) # 데이터 a과 centroid i1, i2, i3 와의 거리를 계산
    
    return distances




