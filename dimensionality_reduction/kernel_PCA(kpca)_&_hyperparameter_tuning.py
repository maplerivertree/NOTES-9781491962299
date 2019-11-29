# Kernal trick can also be applied for dimentionality reduction
# making it possible to perform complex nonlinear projections for dimentionality reduction


from sklearn.decomposition import KernalPCA

rbf_pca = KernelPCA(n_components = 2, kernal = "rbf", gamma = 0.04)
X_reduced = rbf_pca.fit_transform(X)


# there are hyperparameters you can tune for kpca
# below is an example of using grid_search to find optimal hyperparameters

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisiticRegression
from sklearn.pipeline import Pipeline

clf = Pipeline([
  ("kpca", KernelPCA(n_components = 2)),
  ("log_reg", LogisticRegression())
  ])
  
  
param_grid = ({
  "kpca_gamma": np.linspace(0.03, 0.05, 10),
  "kpca_kernel", ["rbf", "sigmoid"]
  })
 
grid_search = GridSearchCV(clf, param_grid, cv= 3)
grid_search.fit(X,y)

print(grid_search.best_params_)
