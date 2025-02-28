import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


class LinearRegressor:
    """
    Extended Linear Regression model with support for categorical variables and gradient descent fitting.
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None

    """
    This next "fit" function is a general function that either calls the *fit_multiple* code that
    you wrote last week, or calls a new method, called *fit_gradient_descent*, not implemented (yet)
    """

    def fit(self, X, y, method="least_squares", learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array).
            y (np.ndarray): Dependent variable data (1D array).
            method (str): method to train linear regression coefficients.
                          It may be "least_squares" or "gradient_descent".
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if method not in ["least_squares", "gradient_descent"]:
            raise ValueError(
                f"Method {method} not available for training linear regression."
            )
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)

        X_with_bias = np.insert(
            X, 0, 1, axis=1
        )  # Adding a column of ones for intercept

        if method == "least_squares":
            self.fit_multiple(X_with_bias, y)
        elif method == "gradient_descent":
            return self.fit_gradient_descent(X_with_bias, y, learning_rate, iterations)

    def fit_simple(self, X, y):
        """
        Fit the model using simple linear regression (one independent variable).

        This method calculates the coefficients for a linear relationship between
        a single predictor variable X and a response variable y.

        Args:
            X (np.ndarray): Independent variable data (1D array).
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if np.ndim(X) > 1:
            X = X.reshape(1, -1)

        # Sacamos las medias de x e y
        x_mean = np.mean(X)
        y_mean = np.mean(y)

        # Sacamos la covarianza de x e y y la varianza de x, obviando n ya que se anula
        numerador = np.sum((X - x_mean) * (y - y_mean))
        denominador = np.sum((X - x_mean) ** 2)
        w = numerador / denominador

        # Calcular la intersección (b)
        b = y_mean - w * x_mean

        # Guardar los valores en los atributos del modelo
        self.coefficients = np.array([w])  # Almacenamos el coeficiente como un array para consistencia
        self.intercept = b

    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        
        # Resolvemos w = (X^T * X)^(-1) * X^T * y
        theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

        # Definimos el intercepto y los coeficientes
        self.intercept = theta[0]  
        self.coefficients = theta[1:] 

    def fit_gradient_descent(self, X, y, learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
      
        # Initialize the parameters to very small values (close to 0)
        m = len(y)
        self.coefficients = (
            np.random.rand(X.shape[1] - 1) * 0.01
        )  # Small random numbers
        self.intercept = np.random.rand() * 0.01
        loss_history = []
        w_history = []

        # Implement gradient descent (TODO)
        for epoch in range(iterations):
            predictions = X.dot(np.r_[self.intercept, self.coefficients])
            error = predictions - y

            # Calculate gradients and update parameters
            gradients = (2 / m) * X.T.dot(error)
            self.intercept -= learning_rate * gradients[0]
            self.coefficients -= learning_rate * gradients[1:]

            w_history.append([self.intercept, *self.coefficients])
            mse = np.mean(error ** 2)
            loss_history.append(mse)

            #Calculate and print the loss every 10 epochs
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}: MSE = {mse}")

        return loss_history, np.array(w_history)


    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).
            fit (bool): Flag to indicate if fit was done.

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """

        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")

        if np.ndim(X) == 1:
            # Al ser unidimensional x se debe convertir en una matriz de una sola columna
            X = X.reshape(-1, 1)
            # Agregamos  una columna de unos para el intercepto
            X_b = np.c_[np.ones((X.shape[0], 1)), X]
            predictions = X_b.dot(np.r_[self.intercept, self.coefficients])    # Calculamos la prediccion

            # Predicción usando la ecuación lineal
        else:
            X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Agregamos una columna de 1s para el intercepto
            predictions = X_b.dot(np.r_[self.intercept, self.coefficients])  # Calculamos la prediccion
        return predictions


def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """
     # R^2 Score
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)  # Variabilidad total de y, TSS
    ss_residual = np.sum((y_true - y_pred) ** 2)  # Suma de errores al cuadrado, RSS
    r_squared = 1 - (ss_residual / ss_total)  # Fórmula de R^2
    #   Pearson unicamente se puede implementar con dos variables, por lo tanto no funcionaría para multiples dimensiones

    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}


def one_hot_encode(X, categorical_indices, drop_first=False):
    """
    One-hot encode the categorical columns specified in categorical_indices. This function
    shall support string variables.

    Args:
        X (np.ndarray): 2D data array.
        categorical_indices (list of int): Indices of columns to be one-hot encoded.
        drop_first (bool): Whether to drop the first level of one-hot encoding to avoid multicollinearity.

    Returns:
        np.ndarray: Transformed array with one-hot encoded columns.
    """
    X_transformed = X.copy()
    for index in sorted(categorical_indices, reverse=True):
        #Extract the categorical column
        categorical_column = X_transformed[:, index]

        # Find the unique categories (works with strings)
        unique_values = np.unique(categorical_column)

        # TODO: Create a one-hot encoded matrix (np.array) for the current categorical column
        one_hot = np.array([(categorical_column == category).astype(int) for category in unique_values]).T

        # Optionally drop the first level of one-hot encoding
        if drop_first:
            one_hot = one_hot[:, 1:]

        #Delete the original categorical column from X_transformed and insert new one-hot encoded columns
        X_transformed = np.delete(X_transformed, index, axis=1)  # Eliminamos la columna original
        X_transformed = np.hstack((X_transformed[:, :index], one_hot, X_transformed[:, index:]))  # Insertamos nuevas columnas


    return X_transformed
