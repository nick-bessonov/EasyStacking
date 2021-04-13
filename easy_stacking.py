import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict


class EasyStacking(BaseEstimator, ClassifierMixin):
    """Стэкинг моделей scikit-learn"""

    def __init__(self, models, meta_model, type_base_models):
        """
        Инициализация
        models - базовые модели для стекинга в виде списка
        ens_model - мета-модель
        type - тип базовых моделей ('regression' | 'classification')
        """
        self.models = models
        self.meta_model = meta_model  # поменять название
        self.type_base_models = type_base_models
        self.n = len(models)  # кол-во базовых моделей
        self.valid = None  # матрица метапризнаков

    def fit(self, X, y, p, cv, err, random_state):
        """
        Обучение стекинга
        p - в каком отношении делить на обучение / тест
            если p = 0 - используем всё обучение!
        cv  (при p=0) - сколько фолдов использовать
        err (при p=0) - величина случайной добавки к метапризнакам
        random_state - инициализация генератора
        """

        # для метода cross_val_predict
        pred_mode = {
            'regression': 'predict',
            'classification': 'predict_proba'}

        if p > 0:  # делим на обучение и тест
            # разбиение на обучение моделей и метамодели
            train, valid, y_train, y_valid = train_test_split(X, y, test_size=p, random_state=random_state)

            # заполнение матрицы для обучения метамодели
            self.valid = np.zeros((valid.shape[0], self.n)) + err * np.random.randn(valid.shape[0], self.n)
            for t, clf in enumerate(self.models):
                clf.fit(train, y_train)
                self.valid[:, t] = clf.predict(valid) if self.type_base_models == 'regression' else \
                    clf.predict_proba(valid)[:, 1]
                # self.valid[:, t] = clf.predict(valid)

            # обучение метамодели
            self.meta_model.fit(self.valid, y_valid)

        else:  # используем всё обучение

            # для регуляризации - добавляем нормальный шум
            self.valid = np.zeros((X.shape[0], self.n)) + err * np.random.randn(X.shape[0], self.n)

            for t, clf in enumerate(self.models):
                # ответы на крос-валидации на каждой модели
                self.valid[:, t] += cross_val_predict(clf, X, y, cv=cv, n_jobs=-1,
                                                      method=pred_mode[self.type_base_models])[:, 1]
                clf.fit(X, y)

            # обучение метамодели
            self.meta_model.fit(self.valid, y)

        return self

    def predict(self, X, y=None):
        """
        Работа стэкинга для регрессора
        """
        # заполение матрицы для мета-классификатора
        X_meta = np.zeros((X.shape[0], self.n))

        for t, clf in enumerate(self.models):
            X_meta[:, t] = clf.predict(X)

        meta_predict = self.meta_model.predict(X_meta)

        return meta_predict

    def predict_proba(self, X, y=None):
        """
        Работа стэкинга для классификатора
        """
        # заполение матрицы для мета-классификатора
        X_meta = np.zeros((X.shape[0], self.n))

        for t, clf in enumerate(self.models):
            X_meta[:, t] = clf.predict_proba(X)[:, 1]

        meta_predict = self.meta_model.predict_proba(X_meta)[:, 1]

        return meta_predict
