import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from io import StringIO


test = tuple(range(2))
a = [test]
print(a)
np.arg

# csv_data = """A,B,C,D
# 1.0,2.0,3.0,4.0
# 5.0,6.0,,8.0
# 10.0,11.0,12.0,

# df = pd.read_csv(StringIO(csv_data))

# imr = SimpleImputer(missing_values=np.nan, strategy="mean")
# imr = imr.fit(df.values)
# data = imr.transform(df.values)
# print(data)

# df = pd.DataFrame(
#     [
#         ["green", "M", 10.1, "class2"],
#         ["red", "L", 13.5, "class1"],
#         ["blue", "XL", 15.3, "class2"],
#     ]
# )
# df.columns = ["color", "size", "price", "classlabel"]
# class_le = LabelEncoder()
# y = class_le.fit_transform(df["classlabel"].values)
# print(y)
# class_mappings = {label: idx for idx, label in enumerate(np.unique(df["classlabel"]))}
# print(type(df["classlabel"]))
# print(class_mappings)

# df_wine = pd.read_csv(
#     "https://archive.ics.uci.edu/ml/" "machine-learning-databases/wine/wine.data",
#     header=None,
# )
# df_wine.columns = [
#     "Class label",
#     "Alcohol",
#     "Malic acid",
#     "Ash",
#     "Alcalinity of ash",
#     "Magnesium",
#     "Total phenols",
#     "Flavanoids",
#     "Nonflavanoid phenols",
#     "Proanthocyanins",
#     "Color intensity",
#     "Hue",
#     "OD280/OD315 of diluted wines",
#     "Proline",
# ]

# X, y = df_wine.iloc[:, 1:].valuesx, df_wine.iloc[:, 0].values

# X_train, y_train, X_test, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=0, stratify=y
# )
