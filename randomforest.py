import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 한글 폰트 설정 (Windows의 경우 한글 폰트를 지정)
rcParams['font.family'] = 'Malgun Gothic'  # Windows에서 Malgun Gothic 사용
rcParams['axes.unicode_minus'] = False  # 마이너스 기호도 제대로 표시

# Load the dataset
data = pd.read_excel('data_preprocessing.xlsx')

# Select features and target
X = data[['스트레스인지율', '우울감경험율', '스트레스로인한정신상담률', '우울증상유병률', '자살생각율', '주관적건강인지율']]
y = data['y']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Get feature importance
feature_importance = model.feature_importances_

# Plot the feature importance
plt.barh(X.columns, feature_importance)
plt.xlabel('Feature Importance')
plt.title('RandomForestRegressor - Feature Importance')
plt.show()


# 변수 중요도를 파일로 저장
importance_df = pd.DataFrame({
    '변수': X.columns,
    '중요도': feature_importance
})

# CSV 파일로 저장 (utf-8 인코딩)
importance_df.to_csv('feature_importance.csv', index=False, encoding='utf-8')
