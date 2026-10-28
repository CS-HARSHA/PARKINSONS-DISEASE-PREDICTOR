"""WP9: Small, deterministic classifier factory for model ablations."""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def make_classifier(name, seed=42):
    name=name.lower()
    if name in ('logistic','logistic_regression'): return LogisticRegression(random_state=seed,max_iter=2000)
    if name in ('linear_svm','svm'): return LinearSVC(random_state=seed)
    if name in ('random_forest','rf'): return RandomForestClassifier(n_estimators=200,max_depth=3,min_samples_leaf=5,random_state=seed)
    if name=='knn': return KNeighborsClassifier(n_neighbors=5)
    raise ValueError(f'Unknown classifier: {name}')
