import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from utilities import visualize_classifier

# Вхідний файл, який містить дані
input_file = 'data_multivar_nb.txt'

# Завантаження даних із вхідного файлу
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Створення наївного байєсовського класифікатора
classifier_new = SVC()

# Тренування класифікатора
classifier_new.fit(X_train, y_train)

# Прогнозування значень для тренувальних даних
y_test_pred = classifier_new.predict(X_test)

# Обчислення якості класифікатора
accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]

# accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print("Accuracy of the new classifier =", round(accuracy, 2), "%")

# Візуалізація результатів роботи класифікатора
visualize_classifier(classifier_new, X_test, y_test)