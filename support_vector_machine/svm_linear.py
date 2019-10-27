from sklearn import datasets
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

raw = datasets.load_iris()
X = raw["data"]
y = (raw["target"] == 2).astype(np.float64)
X = raw["data"][:, (2, 3)]

#LINEAR example
svm_clf = Pipeline([("scaler", StandardScaler()), ("linear_svc", LinearSVC(C = 1, loss = "hinge"))])
svm_clf.fit(X,y)
svm_clf.predict([[5.5,1.7]])
# c above: smaller C value leads to wider steet but more margin violations

