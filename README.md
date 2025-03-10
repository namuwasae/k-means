# K-means 클러스터링 알고리즘

K-means는 비지도 학습 알고리즘으로, 데이터를 K개의 클러스터로 그룹화하는 방법입니다. 이 프로젝트는 NumPy를 사용하여 K-means 알고리즘을 처음부터 구현한 것입니다.

## 목차

- [개요](#개요)
- [알고리즘 설명](#알고리즘-설명)
- [설치 방법](#설치-방법)
- [사용 방법](#사용-방법)
- [파라미터 설명](#파라미터-설명)
- [예제](#예제)
- [프로젝트 구조](#프로젝트-구조)
- [참고 자료](#참고-자료)

## 개요

K-means 알고리즘은 다음과 같은 특징을 가집니다:

- **비지도 학습**: 레이블이 없는 데이터에서 패턴을 찾습니다.
- **분할 클러스터링**: 데이터를 K개의 서로 겹치지 않는 부분집합으로 나눕니다.
- **거리 기반**: 데이터 포인트 간의 유클리드 거리를 사용하여 유사성을 측정합니다.
- **반복적 최적화**: 중심점(centroid)을 반복적으로 업데이트하여 최적의 클러스터를 찾습니다.

## 알고리즘 설명

K-means 알고리즘은 다음 단계로 작동합니다:

1. **초기화**: K개의 중심점(centroid)을 무작위로 선택합니다.
2. **할당**: 각 데이터 포인트를 가장 가까운 중심점에 할당합니다.
3. **업데이트**: 각 클러스터의 중심점을 해당 클러스터에 속한 모든 데이터 포인트의 평균으로 업데이트합니다.
4. **반복**: 중심점이 더 이상 크게 변하지 않거나 최대 반복 횟수에 도달할 때까지 2-3단계를 반복합니다.

## 설치 방법

이 프로젝트를 사용하기 위해 다음 단계를 따르세요:

1. 저장소 클론:
   ```bash
   git clone https://github.com/yourusername/K_means.git
   cd K_means
   ```

2. 가상 환경 생성 및 활성화:
   ```bash
   python -m venv kmeans_env
   source kmeans_env/bin/activate  # Linux/Mac
   # 또는
   kmeans_env\Scripts\activate  # Windows
   ```

3. 필요한 패키지 설치:
   ```bash
   pip install -r requirements.txt
   ```

## 사용 방법

K-means 알고리즘을 사용하는 기본 예제:

```python
from src.kmeans import KMeans
import numpy as np

# 데이터 준비
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

# K-means 모델 초기화
kmeans = KMeans(k=2, max_iter=100, tol=1e-4)

# 모델 학습
kmeans.fit(X)

# 클러스터 결과 확인
print("클러스터 할당:", kmeans.clusters)
print("중심점:", kmeans.centroids)
print("반복 횟수:", kmeans.n_iters_)
```

## 파라미터 설명

`KMeans` 클래스는 다음 파라미터를 받습니다:

- **k** (int): 클러스터의 수
- **max_iter** (int): 최대 반복 횟수
- **tol** (float, 기본값=1e-4): 수렴 판단을 위한 허용 오차. 중심점 변화량이 이 값보다 작으면 알고리즘이 수렴했다고 판단합니다.
  - 값이 클수록 빠르게 수렴하지만 정확도가 떨어질 수 있습니다.
  - 값이 작을수록 더 정밀하게 수렴하지만 계산 시간이 길어질 수 있습니다.
  - 일반적으로 1e-4(0.0001)가 적절한 값입니다.

## 예제

### Iris 데이터셋 클러스터링

```python
from src.kmeans import KMeans
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Iris 데이터 로드
iris = load_iris()
X = iris.data

# K-means 모델 학습
kmeans = KMeans(k=3, max_iter=100)
kmeans.fit(X)

# 결과 시각화 (첫 두 특성만 사용)
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=kmeans.clusters, cmap='viridis', alpha=0.5)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='X', s=100)
plt.title('Iris 데이터셋 K-means 클러스터링 결과')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()
```

## 프로젝트 구조

```
K_means/
├── main.py                 # 메인 실행 파일
├── requirements.txt        # 필요한 패키지 목록
├── README.md               # 프로젝트 설명
├── Iris.csv                # 예제 데이터셋
└── src/                    # 소스 코드
    ├── __init__.py         # 패키지 초기화
    ├── kmeans.py           # K-means 알고리즘 구현
    ├── preprocessing.py    # 데이터 전처리 함수
    └── utils.py            # 유틸리티 함수 (거리 계산 등)
```

## 참고 자료

- [K-means 알고리즘 설명](https://en.wikipedia.org/wiki/K-means_clustering)
- [scikit-learn K-means 문서](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [K-means 시각화 데모](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)
