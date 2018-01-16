# from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib
import pandas as pd


# load data and split sets
df = pd.read_csv("data/features.csv")
print(df.shape)
X = df.iloc[:, :-2].values
Y = df.iloc[:, -1]
# print(Y)
label_encode = LabelEncoder()
Y = label_encode.fit_transform(Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

# Create pipeline and train the model
scaler = StandardScaler()
selector = VarianceThreshold(0.1)
# classifier = GaussianNB()
classifier = svm.SVC(kernel='linear')
pipeline = make_pipeline(scaler, selector, classifier)
pipeline.fit(x_train, y_train)

# Get predictions and accuracy on the test set
y_hat = pipeline.predict(x_test)
print("Accuracy on the test set:")
print(classification_report(y_test, y_hat))

# Save model for future retrieval
joblib.dump(pipeline, "tmp/trained_model.pkl", protocol=2)
print("Model saved to: tmp/trained_model.pkl")
