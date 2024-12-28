# from sklearn.cluster import KMeans
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt
# import pandas as pd

# data = pd.read_excel('data_preprocessing.xlsx')

# # Prepare the features for clustering
# X_cluster = data[['스트레스인지율', '우울감경험율', '스트레스로인한정신상담률', '우울증상유병률', '자살생각율', '주관적건강인지율']]

# # Min-Max scaling
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X_cluster)

# # Using the Elbow method to determine the optimal number of clusters
# inertia = []
# for k in range(1, 11):
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(X_scaled)
#     inertia.append(kmeans.inertia_)

# # Plotting the Elbow Curve
# plt.plot(range(1, 11), inertia, marker='o')
# plt.title('Elbow Method for Optimal K')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Inertia')
# plt.show()

# # After identifying optimal k (e.g., k=3), apply K-Means with that number of clusters
# kmeans = KMeans(n_clusters=3, random_state=42)
# clusters = kmeans.fit_predict(X_scaled)

# # Add cluster labels to the dataframe
# data['Cluster'] = clusters

# # Visualizing the clustering results
# plt.scatter(data['스트레스인지율'], data['우울증상유병률'], c=data['Cluster'], cmap='viridis')
# plt.title('K-Means Clustering Results')
# plt.xlabel('Stress Recognition Rate')
# plt.ylabel('Depression Symptom Prevalence Rate')
# plt.colorbar(label='Cluster')
# plt.show()


# # Example: Check for intersections between clusters
# cluster_intersection = data[data['Cluster'] == 0]  # Regions in cluster 0
# print(cluster_intersection)

# # If you need to compare multiple clusters, you can find intersections by comparing labels
# cluster_intersection = data[(data['Cluster'] == 0) | (data['Cluster'] == 1)]  # Intersection of cluster 0 and 1
# print(cluster_intersection)

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams

data = pd.read_excel('data_preprocessing.xlsx')

rcParams['font.family'] = 'Malgun Gothic'  # Windows에서 Malgun Gothic 사용
rcParams['axes.unicode_minus'] = False  # 마이너스 기호도 제대로 표시

# K-Means 클러스터링을 위한 데이터 준비
# 1. 주관적 건강인지율과 인구밀도 기반 클러스터링
X_cluster1 = data[['주관적건강인지율', '인구밀도']]
scaler = MinMaxScaler()
X_scaled1 = scaler.fit_transform(X_cluster1)
kmeans1 = KMeans(n_clusters=3, random_state=42)
clusters1 = kmeans1.fit_predict(X_scaled1)
data['Cluster1'] = clusters1

# 2. 자살생각율과 인구밀도 기반 클러스터링
X_cluster2 = data[['자살생각율', '인구밀도']]
X_scaled2 = scaler.fit_transform(X_cluster2)
kmeans2 = KMeans(n_clusters=3, random_state=42)
clusters2 = kmeans2.fit_predict(X_scaled2)
data['Cluster2'] = clusters2

# 3. 주관적 건강인지율과 자살생각율 기반 클러스터링
X_cluster3 = data[['주관적건강인지율', '자살생각율']]
X_scaled3 = scaler.fit_transform(X_cluster3)
kmeans3 = KMeans(n_clusters=3, random_state=42)
clusters3 = kmeans3.fit_predict(X_scaled3)
data['Cluster3'] = clusters3

# 클러스터 시각화 (각 클러스터를 색깔로 구분)
plt.scatter(data['주관적건강인지율'], data['자살생각율'], c=data['Cluster1'], cmap='viridis')
plt.title('K-Means Clustering (주관적 건강인지율 vs 자살생각율)')
plt.xlabel('주관적 건강인지율')
plt.ylabel('자살생각율')
plt.colorbar(label='Cluster')
plt.show()
