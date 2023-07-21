
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogReg:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):

        X = np.array(X) # переведем в numpy для матричных преобразований
        y = np.array(y)
        n_samples, n_features = X.shape
    
        self.coef_ = np.random.normal(size=X.shape[1])
        self.intercept_ = np.random.normal()

        BSELoss_old = self.bseloss(X, y)
        
        # Градиентный спуск для обновления весов и свободного члена
        while True:

        #for _ in range(1):
            # Вычисляем линейную комбинацию весов и данных
            z = np.dot(X, self.coef_) + self.intercept_

            # Применяем сигмоидную функцию для получения предсказаний
            y_pred = sigmoid(z)

            error = y_pred - y

            # Вычисляем градиенты функции потерь
            gradient_coef = np.dot(X.T, error) / n_features
            gradient_intercept = np.sum(error) / n_features

            # Обновляем веса с помощью градиентного спуска
            self.coef_ -= self.learning_rate * gradient_coef
            self.intercept_ -= self.learning_rate * gradient_intercept


            BSELoss = - (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)).mean()

            if BSELoss_old < BSELoss:
                break
            else:
                BSELoss_old = BSELoss

        return self.coef_, self.intercept_

    def predict(self, X_new):
        # Вычисляем линейную комбинацию весов и новых данных
        z = np.dot(X_new, self.coef_) + self.intercept_

        # Применяем сигмоидную функцию для получения вероятности принадлежности к классу 1
        y_pred = sigmoid(z)

        # Применяем порог 0.5 для определения метки класса (0 или 1)
        predictions = (y_pred >= 0.5).astype(int)

        return predictions

    def bseloss(self, X, y):

        z = np.dot(X, self.coef_) + self.intercept_
        y_pred = sigmoid(z)
        BSELoss = - (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)).mean()
        
        return BSELoss