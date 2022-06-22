# Datasets:
# importing diabetes dataset from sklearn
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_wine
from sklearn.datasets import load_iris

import pandas as pd

# Loading iris dataset
data = load_iris()

# storing as data frame
dataframe = pd.DataFrame(data.data, columns=data.feature_names)

# Convert entire data frame as string and print
print(dataframe.to_string())