import numpy as np
from utils import compute_distance

# KMeans 클래스 정의
class KMeans:
    # init
    def __init__(self, k, max_iter):
        self.k = k                # cluster 개수 (k)
        self.max_iter = max_iter  # 최대 반복 횟수(I)
        self.centroids = None
        self.clusters = None
        self.inertia_ = None  # 클러스터 내 분산
        self.n_iters_ = None  # 실제 반복 횟수
        self.centroid_history = []  # 중심점 변화 기록
    
    
    def fit(self, data):
        self.centroids = self.init_centroid(data)
        self.centroid_history = [self.centroids.copy()]

        for i in range(self.max_iter):
            # 거리 계산
            distances = compute_distance(data, self.centroids)
            # 클러스터 할당
            clusters = self.assign_clusters(distances)
            # 중심점 업데이트
            new_centroids = self.update_centroids(data, self.centroids, clusters)
            self.centroid_history.append(new_centroids.copy())
            
            ##############################################################################
            # 중심점 수렴하는지 확인
            centroid_shift = np.sum((new_centroids - self.centroids) ** 2)
            if centroid_shift < self.tol:
                self.n_iters_ = i + 1
                print(f"수렴 완료: {self.n_iters_}번째 반복")
                break
            
            self.centroids = new_centroids
            
    
    # save history
    def get_centroid_history(self):
        return self.save_centroid_history

    # Centroid값 임의 설정(무작위 설정)
    ## choice 함수를 이용해 존재하는 데이터중 하나를 무작위로 추출해 centroids 배열에 저장.
    def init_centroid(self, data):
        
        x = np.random.choice(data.shape[0], size = self.k, replace = False) # 인덱스 무작위
        centroids=data[x]       # 인덱스에 해당하는 행데이터(초기 centroid) 추출
        return centroids
    
    
    # 데이터 클러스터링
    ## 지금 distances의 행은 데이터 포인트, 열은 centroid 이므로 각 데이터 포인트와 각 centroid 사이의 거리를 나타냄.
    ## 각 데이터 포인트와 가장 가까운 centroid의 인덱스를 반환. 이걸 fit에서 반복문으로 실행해주면 실행해 clusters에 저장해줄 것임.
    def assign_clusters(distances):
        return np.argmin(distances, axis=1)
        
    
    # 각 클러스터에 할당된 데이터 포인터들의 평균을 계산해 새 중심점을 반환.
    def update_centroids(data, clusters, k):
        n_features=data.shape[1]        
        new_centroids = np.zeros((k,n_features))    # 새 중심점의 feature 개수는 data의 feature 개수와 같다.

        for i in range(k):
            new_centroids[i] = np.mean(data[clusters==i], axis = 0) # 같은 클러스터에 속한 데이터들의 평균을 구해서 새 중심점으로 정하는 부분.

        return new_centroids        

