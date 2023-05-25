import json
import pickle
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


class ChurnLinearModel(BaseEstimator):
    def __init__(self):
        super(ChurnLinearModel, self).__init__()
        self.model = LogisticRegression()
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save(self, filename):
        pickle.dump({
            'scaler': self.scaler,
            'model': self.model,
        }, open(filename, 'wb'))

    def load(self, filename):
        checkpoint = pickle.load(open(filename, 'rb'))
        self.scaler = checkpoint['scaler']
        self.model = checkpoint['model']


class ChurnDecisionTree(BaseEstimator):
    def __init__(self, max_depth=None, criterion='gini'):
        super(ChurnDecisionTree, self).__init__()
        self.max_depth = max_depth
        self.criterion = criterion
        self.model = DecisionTreeClassifier(
            max_depth=max_depth, criterion=criterion,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, filename):
        json_filename = filename.split('.')[0] + '.json'
        json.dump(self.get_params(), open(json_filename, 'w'))
        pickle.dump({
            'model': self.model,
        }, open(filename, 'wb'))

    def load(self, filename):
        json_filename = filename.split('.')[0] + '.json'
        params = json.load(open(json_filename, 'r'))
        self.set_params(**params)
        checkpoint = pickle.load(open(filename, 'rb'))
        self.model = checkpoint['model']
