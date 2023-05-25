import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

# Завантаження даних
input_file = 'data_regr_1.txt'
data = np.loadtxt(input_file, delimiter=',')
X, Y = data[:, :-1], data[:, -1]

# Розділення даних на навчальні та тестові
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Тренувальні дані
X_train, Y_train = X[:num_training], Y[:num_training]
# Тестові дані
X_test, Y_test = X[num_training:], Y[num_training:]

# Створення лінійної регресії
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, Y_train)

# Прогнозування результатів
Y_test_pred = linear_regressor.predict(X_test)

# Побудова графіка
plt.scatter(X_test, Y_test, color='green')
plt.plot(X_test, Y_test_pred, color='black', linewidth=4)
plt.xticks(())
plt.yticks(())
plt.show()

# Виведення результатів
print("Linear regressor performance:")
print(f"Mean absolute error = {round(sm.mean_absolute_error(Y_test, Y_test_pred), 2)}")
print(f"Mean squared error = {round(sm.mean_squared_error(Y_test, Y_test_pred), 2)}")
print(f"Median absolute error = {round(sm.median_absolute_error(Y_test, Y_test_pred), 2)}")
print(f"Explain variance score = {round(sm.explained_variance_score(Y_test, Y_test_pred), 2)}")
print(f"R2 score = {round(sm.r2_score(Y_test, Y_test_pred), 2)}")