#### python
# filepath: c:\Users\sayus\OneDrive\Desktop\Qfs\qFS\test.py

"""
This script tests the feature selection methods implemented by qFS_module.
"""

from qFS.qFS import qFS, qFSK, qFSiP, get_i
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import time
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# Increase the max_iter to reduce convergence warnings for LogisticRegression
logistic_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(solver='lbfgs', max_iter=3000))
])

classifiers = [
    ('DecisionTree', DecisionTreeClassifier(), {'max_depth': [5, 10, 15]}),
    ('LogisticRegression', logistic_pipeline, {'logreg__C': [0.1, 1, 10]})
]

dataset = load_digits()
n_features = dataset.data.shape[1]
npieces = get_i(n_features)

for tag, clf, param_grid in classifiers:
    """
    No Feature Selection
    """
    grid = GridSearchCV(clf, param_grid, cv=10, scoring='accuracy')
    grid.fit(dataset.data, dataset.target)
    
    print("No Feature Selection")
    print("Classifier: {}".format(tag))
    print("Best score: {}\n".format(grid.best_score_))
    
    """
    qFS
    """
    qfs_selector = qFS()
    t0 = time.time()
    qfs_selector.fit(dataset.data, dataset.target)
    elapsed_t = time.time() - t0

    # Validation
    grid = GridSearchCV(clf, param_grid, cv=10, scoring='accuracy')
    grid.fit(dataset.data[:, qfs_selector.idx_sel], dataset.target)

    print("qFS")
    print("Classifier: {}".format(tag))
    print("Best score: {}".format(grid.best_score_))
    print("Elapsed Time: {}\n".format(elapsed_t))

    k = len(qfs_selector.idx_sel)  # Number of selected features for qFSK and qFSiP

    """
    qFS#
    """
    qfsk_selector = qFSK(k=k)
    t0 = time.time()
    qfsk_selector.fit(dataset.data, dataset.target)
    elapsed_t = time.time() - t0

    # Validation
    grid = GridSearchCV(clf, param_grid, cv=10, scoring='accuracy')
    grid.fit(dataset.data[:, qfsk_selector.idx_sel], dataset.target)

    print("qFS#")
    print("Classifier: {}".format(tag))
    print("Best score: {}".format(grid.best_score_))
    print("Elapsed Time: {}\n".format(elapsed_t))

    """
    FCBiP (qFSiP)
    """
    for i in npieces:
        qfsip_selector = qFSiP(npieces=i, k=k)
        t0 = time.time()
        qfsip_selector.fit(dataset.data, dataset.target)
        elapsed_t = time.time() - t0

        # Validation
        grid = GridSearchCV(clf, param_grid, cv=10, scoring='accuracy')
        grid.fit(dataset.data[:, qfsip_selector.idx_sel], dataset.target)

        print("qFSiP with {} pieces".format(i))
        print("Classifier: {}".format(tag))
        print("Best score: {}".format(grid.best_score_))
        print("Elapsed Time: {}\n".format(elapsed_t))
"""
OUTPUT

No Feature Selection
Classifer: DecisionTree
Best score: 0.836393989983

qFS
Classifer: DecisionTree
Best score: 0.82081246522
Elapsed Time: 1.53129601479

qFS#
Classifer: DecisionTree
Best score: 0.823594880356
Elapsed Time: 1.55748701096

qFSiP with 2 pieces
Classifer: DecisionTree
Best score: 0.827490261547
Elapsed Time: 2.3456659317

qFSiP with 4 pieces
Classifer: DecisionTree
Best score: 0.797996661102
Elapsed Time: 1.23591303825

qFSiP with 8 pieces
Classifer: DecisionTree
Best score: 0.820255982193
Elapsed Time: 0.638503074646

qFSiP with 16 pieces
Classifer: DecisionTree
Best score: 0.816917084029
Elapsed Time: 0.343441963196

qFSiP with 32 pieces
Classifer: DecisionTree
Best score: 0.821925431274
Elapsed Time: 0.196324110031

No Feature Selection
Classifer: LogisticRegression
Best score: 0.936004451864

qFS
Classifer: LogisticRegression
Best score: 0.903171953255
Elapsed Time: 1.5462949276

qFS#
Classifer: LogisticRegression
Best score: 0.903171953255
Elapsed Time: 1.56521987915

qFSiP with 2 pieces
Classifer: LogisticRegression
Best score: 0.875904284919
Elapsed Time: 2.38653302193

qFSiP with 4 pieces
Classifer: LogisticRegression
Best score: 0.894268224819
Elapsed Time: 1.23389911652

qFSiP with 8 pieces
Classifer: LogisticRegression
Best score: 0.884251530328
Elapsed Time: 0.643393039703

qFSiP with 16 pieces
Classifer: LogisticRegression
Best score: 0.90428491931
Elapsed Time: 0.346354961395

qFSiP with 32 pieces
Classifer: LogisticRegression
Best score: 0.903728436283
Elapsed Time: 0.195168018341

"""