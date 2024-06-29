import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np


class LinearRegressionGD:
    def __init__(self, eta=0.01, epochs=50, random_state=1) -> None:
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.0)
        self.loss_ = []

        for _ in range(self.epochs):
            output = self.net_input(X)
            error = y - output
            self.w_ += self.eta * 2 * X.T.dot(error) / X.shape[0]
            self.b_ += self.eta * 2 * error.mean()
            loss = (error**2).mean()
            self.loss_.append(loss)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return self.net_input(X)


def lin_regplot(X, y, model):
    plt.scatter(X, y, c="steelblue", edgecolors="white", s=70)
    r = model.predict(X)
    plt.plot(X, r, color="black", lw=2)


columns = [
    "Overall Qual",
    "Overall Cond",
    "Gr Liv Area",
    "Central Air",
    "Total Bsmt SF",
    "SalePrice",
]

df = pd.read_csv(
    "http://jse.amstat.org/v19n3/decock/AmesHousing.txt", sep="\t", usecols=columns
)

df["Central Air"] = df["Central Air"].map({"N": 0, "Y": 1})
df = df.dropna(axis=0)

X = df[["Gr Liv Area"]].values
y = df["SalePrice"].values
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

lg = LinearRegressionGD(eta=0.1)
lg.fit(X_std, y_std)

slr = LinearRegression()
slr.fit(X, y)
y_pred = slr.predict([[2500]])

print(y_pred[0])
# lin_regplot(X_std, y_std, lg)
# plt.xlabel(" Living area above ground (standardized)")
# plt.ylabel("Sale price (standardized)")
# plt.show()

# plt.plot(range(1, lg.epochs + 1), lg.loss_)
# plt.ylabel("MSE")
# plt.xlabel("Epoch")
# plt.show()
