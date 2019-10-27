from sklearn import datasets
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC

#Non-linear

X, y = datasets.make_moons(n_samples=300, noise=0.08, shuffle=False)

polynominal_svm_clf = Pipeline([("poly_features", PolynomialFeatures(degree = 3)), ("scaler", StandardScaler()), ("svm_clf", LinearSVC(C = 10, loss ="hinge"))])
polynominal_svm_clf.fit(X, y)
