from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ("o", "s", "^", "v", "<")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    # xx1 = x-values of the z (lab) coordinates
    # xx2 = y-values of the z coordinates?
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    # plt.xlim(xx1.min(), xx1.max())
    # plt.ylim(xx2.min(), xx2.max())
    # plot class examples
    # for idx, cl in enumerate(np.unique(y)):
    #     plt.scatter(
    #         x=X[y == cl, 0],
    #         y=X[y == cl, 1],
    #         alpha=0.8,
    #         c=colors[idx],
    #         marker=markers[idx],
    #         label=f"Class {cl}",
    #         edgecolor="black",
    #     )
    # # highlight test examples
    # if test_idx:
    #     # plot all examples
    #     X_test, y_test = X[test_idx, :], y[test_idx]

    #     plt.scatter(
    #         X_test[:, 0],
    #         X_test[:, 1],
    #         c="none",
    #         edgecolor="black",
    #         alpha=1.0,
    #         linewidth=1,
    #         marker="o",
    #         s=100,
    #         label="Test set",
    #     )


class LogisticRegressionGD:
    def __init__(self, eta=0.01, n_iter=10, random_state=1):
        self.eta = eta
        self.n_iter = 10
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.0)
        self.losses_ = []

        for _ in range(self.n_iter):
            output = self.activation(self.net_input(X))
            errors = y - output
            # don't get why we have to multiply by 2 here?
            self.w_ = self.eta * 2 * X.T.dot(errors) / X.shape[0]
            self.b_ = self.eta * 2 * errors.mean()
            loss = -y.dot(np.log(output)) - (
                (1 - y).dot(np.log(1 - output)) / X.shape[0]
            )
            self.losses_.append(loss)

        return self

    def activation(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.5, 1, 0)


iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=1,
    stratify=y,
    # stratification means that this method will return training and test subsets that have
    # the same proportions of class labels as the input dataset
)

# what is the purpose of standardizing?
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric="minkowski")
knn.fit(X_train_std, y_train)
plot_decision_regions(
    X_combined_std, y_combined, classifier=knn, test_idx=range(105, 150)
)
plt.xlabel("Petal length [standardized]")
plt.ylabel("Petal width [standardized]")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()

# lr = LogisticRegression(C=100.0, solver="lbfgs")
# lr.fit(X_train_std, y_train)
# print(lr.predict_proba(X_test_std[:3, :]).arg(axis=1))
# plot_decision_regions(
#     X_combined_std, y_combined, classifier=lr, test_idx=range(105, 150)
# )
# plt.xlabel("Petal length [standardized]")
# plt.ylabel("Petal width [standardized]")
# plt.legend(loc="upper left")
# plt.tight_layout()
plt.show()


# ppn = Perceptron(eta0=0.1, random_state=1)
# # ppn.fit(X_train_std, y_train)
# # y_pred = ppn.predict(X_test_std)
# # print("missed: %d" % (y_test != y_pred).sum())

# # X_combined_std = np.vstack((X_train_std, X_test_std))
# # y_combined = np.hstack((y_train, y_test))
# # plot_decision_regions(
# #     X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150)
# # )
# # plt.xlabel("Petal length [standardized]")
# # plt.ylabel("Petal width [standardized]")
# # plt.legend(loc="upper left")
# # plt.tight_layout()
# # plt.show()
