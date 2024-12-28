from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import font_manager

# 한글 폰트 설정 (macOS, Windows, Linux에 따라 다를 수 있습니다)
plt.rcParams['font.family'] = 'NanumGothic'  # 나눔고딕을 사용
# 혹은 'Malgun Gothic' 등 시스템에 맞는 폰트를 사용할 수 있습니다.

# 확인: 한글 폰트 적용 여부 확인
print(font_manager.findSystemFonts(fontpaths=None, fontext='ttf'))


# 데이터 불러오기
data = pd.read_excel('data_preprocessing.xlsx')

# Min-Max Scaling
scaler = MinMaxScaler()

# -------------------------- 1. 주관적 건강 인지율과 노인 인구 밀도 기준 클러스터링 --------------------------
# 주관적 건강 인지율과 노인 인구 밀도를 선택
X_cluster1 = data[['주관적건강인지율', '65세이상고령자수']]

# 데이터 스케일링
X_scaled1 = scaler.fit_transform(X_cluster1)

# Elbow Method로 최적의 클러스터 개수 찾기
inertia1 = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled1)
    inertia1.append(kmeans.inertia_)

# Elbow Curve 시각화
plt.plot(range(1, 11), inertia1, marker='o')
plt.title('Elbow Method for 주관적 건강 인지율과 65세이상 고령자수')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.show()

# K-Means 클러스터링 실행
kmeans1 = KMeans(n_clusters=3, random_state=42)
clusters1 = kmeans1.fit_predict(X_scaled1)

# 클러스터 라벨을 데이터프레임에 추가
data['Cluster1'] = clusters1

# 클러스터링 결과 시각화
plt.scatter(data['주관적건강인지율'], data['65세이상고령자수'], c=data['Cluster1'], cmap='viridis')
plt.title('K-Means Clustering (주관적 건강 인지율 vs 65세이상 고령자수)')
plt.xlabel('주관적 건강인지율')
plt.ylabel('65세이상 고령자수')
plt.colorbar(label='Cluster')
plt.show()

# -------------------------- 2. 노인 인구 밀도와 자살 생각률 기준 클러스터링 --------------------------
# 노인 인구 밀도와 자살 생각률을 선택
X_cluster2 = data[['65세이상고령자수', '자살생각율']]

# 데이터 스케일링
X_scaled2 = scaler.fit_transform(X_cluster2)

# Elbow Method로 최적의 클러스터 개수 찾기
inertia2 = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled2)
    inertia2.append(kmeans.inertia_)

# Elbow Curve 시각화
plt.plot(range(1, 11), inertia2, marker='o')
plt.title('Elbow Method for 65세이상 고령자수와 자살 생각률')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.show()

# K-Means 클러스터링 실행
kmeans2 = KMeans(n_clusters=3, random_state=42)
clusters2 = kmeans2.fit_predict(X_scaled2)

# 클러스터 라벨을 데이터프레임에 추가
data['Cluster2'] = clusters2

# 클러스터링 결과 시각화
plt.scatter(data['65세이상고령자수'], data['자살생각율'], c=data['Cluster2'], cmap='viridis')
plt.title('K-Means Clustering (65세이상 고령자수 vs 자살 생각률)')
plt.xlabel('65세이상 고령자수')
plt.ylabel('자살생각율')
plt.colorbar(label='Cluster')
plt.show()

# -------------------------- 3. 주관적 건강 인지율과 자살 생각률 기준 클러스터링 --------------------------
# 주관적 건강 인지율과 자살 생각률을 선택
X_cluster3 = data[['주관적건강인지율', '자살생각율']]

# 데이터 스케일링
X_scaled3 = scaler.fit_transform(X_cluster3)

# Elbow Method로 최적의 클러스터 개수 찾기
inertia3 = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled3)
    inertia3.append(kmeans.inertia_)

# Elbow Curve 시각화
plt.plot(range(1, 11), inertia3, marker='o')
plt.title('Elbow Method for 주관적 건강 인지율과 자살 생각률')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.show()

# K-Means 클러스터링 실행
kmeans3 = KMeans(n_clusters=3, random_state=42)
clusters3 = kmeans3.fit_predict(X_scaled3)

# 클러스터 라벨을 데이터프레임에 추가
data['Cluster3'] = clusters3

# 클러스터링 결과 시각화
plt.scatter(data['주관적건강인지율'], data['자살생각율'], c=data['Cluster3'], cmap='viridis')
plt.title('K-Means Clustering (주관적 건강 인지율 vs 자살생각율)')
plt.xlabel('주관적 건강인지율')
plt.ylabel('자살생각율')
plt.colorbar(label='Cluster')
plt.show()
