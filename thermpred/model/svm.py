import sklearn

from . import _sklearn_model


class SupportVectorMachine(_sklearn_model.SklearnModel):
    """
    Implementation of a class for Support Vector Machine respective Regression.

    See :obj:`~thermpred.model._base_model.BaseModel` for more information on the attributes.
    """
    featureset = 'features'

    def define_model(self):
        """
        Definition of the actual prediction model.

        See :obj:`~thermpred.model._base_model.BaseModel` for more information.
        """
        kernel = self.suggest_hyperparam_to_optuna('kernel')
        reg_c = self.suggest_hyperparam_to_optuna('C')
        if kernel == 'poly':
            degree = self.suggest_hyperparam_to_optuna('degree')
            gamma = self.suggest_hyperparam_to_optuna('gamma')
        elif kernel in ['rbf', 'sigmoid']:
            degree = 42  # default
            gamma = self.suggest_hyperparam_to_optuna('gamma')
        elif kernel == 'linear':
            degree = 42  # default
            gamma = 42  # default
        if self.task == 'classification':
            return sklearn.svm.SVC(kernel=kernel, C=reg_c, degree=degree, gamma=gamma, random_state=42,
                                   max_iter=1000000, probability=True)
        else:
            return sklearn.svm.SVR(kernel=kernel, C=reg_c, degree=degree, gamma=gamma, max_iter=100000000)

    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~thermpred.model._base_model.BaseModel` for more information on the format.
        """
        return {
            'kernel': {
                'datatype': 'categorical',
                'list_of_values': ['linear', 'poly', 'rbf'],
            },
            'degree': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 5
            },
            'gamma': {
                'datatype': 'float',
                'lower_bound': 10**-3,
                'upper_bound': 10**3,
                'log': True
            },
            'C': {
                'datatype': 'float',
                'lower_bound': 10**-3,
                'upper_bound': 10**3,
                'log': True
            }
        }
