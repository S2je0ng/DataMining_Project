from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import font_manager

plt.rcParams['font.family'] = 'NanumGothic'

print(font_manager.findSystemFonts(fontpaths=None, fontext='ttf'))

data = pd.read_excel('data_preprocessing.xlsx')

scaler = MinMaxScaler()

# 주관적 건강 인지율과 노인 인구 밀도
X_cluster1 = data[['주관적건강인지율', '65세이상고령자수']]

X_scaled1 = scaler.fit_transform(X_cluster1)

inertia1 = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled1)
    inertia1.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia1, marker='o')
plt.title('Elbow Method for 주관적 건강 인지율과 65세이상 고령자수')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.show()

kmeans1 = KMeans(n_clusters=3, random_state=42)
clusters1 = kmeans1.fit_predict(X_scaled1)

data['Cluster1'] = clusters1

# 클러스터링 결과
plt.scatter(data['주관적건강인지율'], data['65세이상고령자수'], c=data['Cluster1'], cmap='viridis')
plt.title('K-Means Clustering (주관적 건강 인지율 vs 65세이상 고령자수)')
plt.xlabel('주관적 건강인지율')
plt.ylabel('65세이상 고령자수')
plt.colorbar(label='Cluster')
plt.show()

# 노인 인구 밀도와 자살 생각률
X_cluster2 = data[['65세이상고령자수', '자살생각율']]

X_scaled2 = scaler.fit_transform(X_cluster2)

inertia2 = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled2)
    inertia2.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia2, marker='o')
plt.title('Elbow Method for 65세이상 고령자수와 자살 생각률')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.show()

kmeans2 = KMeans(n_clusters=4, random_state=42)
clusters2 = kmeans2.fit_predict(X_scaled2)

data['Cluster2'] = clusters2

# 클러스터링 결과
plt.scatter(data['65세이상고령자수'], data['자살생각율'], c=data['Cluster2'], cmap='viridis')
plt.title('K-Means Clustering (65세이상 고령자수 vs 자살 생각률)')
plt.xlabel('65세이상 고령자수')
plt.ylabel('자살생각율')
plt.colorbar(label='Cluster')
plt.show()

# 주관적 건강 인지율과 자살 생각률
X_cluster3 = data[['주관적건강인지율', '자살생각율']]

X_scaled3 = scaler.fit_transform(X_cluster3)

inertia3 = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled3)
    inertia3.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia3, marker='o')
plt.title('Elbow Method for 주관적 건강 인지율과 자살 생각률')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.show()

kmeans3 = KMeans(n_clusters=4, random_state=42)
clusters3 = kmeans3.fit_predict(X_scaled3)

data['Cluster3'] = clusters3

# 클러스터링 결과
plt.scatter(data['주관적건강인지율'], data['자살생각율'], c=data['Cluster3'], cmap='viridis')
plt.title('K-Means Clustering (주관적 건강 인지율 vs 자살생각율)')
plt.xlabel('주관적 건강인지율')
plt.ylabel('자살생각율')
plt.colorbar(label='Cluster')
plt.show()
