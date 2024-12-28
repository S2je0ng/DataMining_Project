import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

rcParams['font.family'] = 'Malgun Gothic'
rcParams['axes.unicode_minus'] = False

data = pd.read_excel('data_preprocessing.xlsx')


X = data[['스트레스인지율', '우울감경험율', '스트레스로인한정신상담률', '우울증상유병률', '자살생각율', '주관적건강인지율']]
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

feature_importance = model.feature_importances_

plt.barh(X.columns, feature_importance)
plt.xlabel('Feature Importance')
plt.title('RandomForestRegressor - Feature Importance')
plt.show()

importance_df = pd.DataFrame({
    '변수': X.columns,
    '중요도': feature_importance
})

importance_df.to_csv('feature_importance.csv', index=False, encoding='utf-8')
