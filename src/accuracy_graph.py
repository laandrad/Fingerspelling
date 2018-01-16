from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
import matplotlib.pyplot as plt


# load data and split sets
df = pd.read_csv("data/features.csv")
X = df.iloc[:, :-2].values
Y = df.iloc[:, -1]

label_encode = LabelEncoder()
Y = label_encode.fit_transform(Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

# Create pipeline and train the model
scaler = StandardScaler()
selector = VarianceThreshold(0.1)
classifier1 = GaussianNB()
pipeline1 = make_pipeline(scaler, selector, classifier1)
classifier2 = svm.SVC(kernel='linear')
pipeline2 = make_pipeline(scaler, selector, classifier2)

nb_acc = []
svm_acc = []
n_iterations = range(100, x_train.shape[0], 100)

for i in n_iterations:
    # Get predictions and accuracy on the test set
    print("computing prediction with {} examples".format(i))

    pipeline1.fit(x_train[:i, ], y_train[:i, ])
    y_hat = pipeline1.predict(x_test)
    acc = accuracy_score(y_test, y_hat)
    nb_acc.append(acc)

    pipeline2.fit(x_train[:i, ], y_train[:i, ])
    y_hat = pipeline2.predict(x_test)
    acc = accuracy_score(y_test, y_hat)
    svm_acc.append(acc)

plt.plot(n_iterations, nb_acc, "r", n_iterations, svm_acc, "b")
plt.text(2000, 0.875, "Naive Bayes Classifier")
plt.text(3000, 0.985, "SVM Classifier")
plt.show()
