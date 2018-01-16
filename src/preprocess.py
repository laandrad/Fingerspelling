import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class CleanData:
    def __init__(self, n_categories):
        self.n_categories = n_categories

    def pre_process(self, x, y, dummy=False):
        # Dummy code outcome variable
        label_encode = LabelEncoder()
        y = label_encode.fit_transform(y)

        if dummy:
            dummies = OneHotEncoder(n_values=self.n_categories, categorical_features=[0])
            y = np.reshape(y, (-1, 1))
            y = dummies.fit_transform(y).toarray()
        else:
            y = y.reshape(y.shape[0], -1)

        return x_train, x_test, y_train, y_test

    def pre_process_features(self, x, scale=True, near_zero=True):
        # Scale variables
        scaler = StandardScaler()
        x = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

