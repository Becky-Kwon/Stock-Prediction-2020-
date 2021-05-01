from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
import pickle
import pandas as pd
import numpy as np
import shap

class RFTuner():
    # initiliazing with loading datset to member variable
    def __init__(self, traindf: pd.DataFrame, target_name: str):
        self.train_df = traindf.copy(deep=True)

        self.loss = 0
        self.X_train, self.X_valid, self.y_train, self.y_valid = None, None, None, None
        self.model = None
        self.explainer = None
        self.shap_values = None

        self.split_data(target_name)

    def split_data(self, target, test_size=0.2):
        df = self.train_df.copy(deep=True)
        y = df.pop(target)
        x = df
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(x, y, test_size=test_size, random_state=32)

    def treeModel(self, max_depth, min_samples_split, min_samples_leaf, n_estimators):
        params = {}
        params['max_depth'] = int(round(max_depth))
        params['min_samples_split'] = min_samples_split
        params['min_samples_leaf'] = min_samples_leaf
        params['n_estimators'] = int(n_estimators)
        params['random_state'] = 32
        params['class_weight'] = 'balanced'

        rf = RandomForestClassifier(**params)
        rf.fit(self.X_train, self.y_train)

        pred = rf.predict(self.X_valid)
        result = mean_squared_error(self.y_valid, pred)
        return -result

    def train_model(self, pbounds, n_iter=15, verbose=2, init_points=5):

        '''pbounds = {
            'max_depth': (5, 30),
            'min_samples_split': (0.1, 0.5),
            'min_samples_leaf': (0.1, 0.25),
            'n_estimators': (10, 100)
        }'''

        # pbouns에 해당하는 지표들의 인자를 택하여 가장 좋은 결과를 낸 값을 선정
        brf = BayesianOptimization(f=self.treeModel, pbounds=pbounds, verbose=verbose)
        brf.maximize(init_points=init_points, n_iter=n_iter)
        print('best_target_value:', brf.max['target'])

        # 위의 과정에서 구한 최적의 패러미터 대입
        params = {}
        params['max_depth'] = int(round(brf.max['params']['max_depth']))
        params['min_samples_split'] = brf.max['params']['min_samples_split']
        params['min_samples_leaf'] = brf.max['params']['min_samples_leaf']
        params['n_estimators'] = int(brf.max['params']['n_estimators'])
        params['random_state'] = 32
        params['class_weight'] = 'balanced'


        # 최적의 패러미터로 만들어진 모델을 학습시킴
        print("Training with best parameters ... ")
        fitted_model = RandomForestClassifier(**params)
        fitted_model.fit(self.X_train, self.y_train)
        pred = fitted_model.predict(self.X_valid)

        self.model = fitted_model
        self.loss = mean_squared_error(self.y_valid, pred)
        self.params = self.model.get_params()

        try:
            self.save_param()

        except:
            print("Failed to save params")

        print('\nRMSE: ', np.sqrt(self.loss))

    def predict(self, test_df):
        preds = self.model.predict(test_df)
        return preds

    def save_param(self, _params=None):
        name = 'rf_' + str(round(self.loss, 4)) + '.params'

        if _params is None:
            params = self.params
        else:
            params = _params

        with open(name, 'wb') as f:
            pickle.dump(params, f, 0)

        print("parameter file " + name + "'' is saved successfully")

    def load_param(self, name):
        with open(name, 'rb') as f:
            return pickle.load(f)

    def load_model(self, param_name):
        params = self.load_param(param_name)
        self.model = RandomForestClassifier(**params)
        self.model.fit(self.X_train, self.y_train)
        pred = self.model.predict_proba(self.X_valid)

        print(self.model)

    def get_shap_values(self):
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(self.X_valid)
        shap.summary_plot(self.shap_values, self.X_valid)
        return self.explainer, self.shap_values