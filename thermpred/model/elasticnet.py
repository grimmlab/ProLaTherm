import sklearn

from . import _sklearn_model


class LogisticRegressionElasticNet(_sklearn_model.SklearnModel):
    """
    Implementation of a class for Linear respective Logistic Regression using ElasticNet penalty.

    See :obj:`~thermpred.model._base_model.BaseModel` for more information on the attributes.
    """
    featureset = 'features'

    def define_model(self):
        """
        Definition of the actual prediction model.

        See :obj:`~thermpred.model._base_model.BaseModel` for more information.
        """
        # Penalty term is fixed to l1, but might also be optimized
        penalty = 'elasticnet'  # self.suggest_hyperparam_to_optuna('penalty')
        if penalty == 'l1':
            l1_ratio = 1
        elif penalty == 'l2':
            l1_ratio = 0
        else:
            l1_ratio = self.suggest_hyperparam_to_optuna('l1_ratio')
        if self.task == 'classification':
            reg_c = self.suggest_hyperparam_to_optuna('C')
            return sklearn.linear_model.LogisticRegression(penalty=penalty, C=reg_c, solver='saga',
                                                           l1_ratio=l1_ratio if penalty == 'elasticnet' else None,
                                                           max_iter=10000, random_state=42, n_jobs=-1)
        else:
            alpha = self.suggest_hyperparam_to_optuna('alpha')
            return sklearn.linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=42)

    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~thermpred.model._base_model.BaseModel` for more information on the format.
        """
        return {
            'penalty': {
                'datatype': 'categorical',
                'list_of_values': ['l1', 'l2', 'elasticnet']
            },
            'l1_ratio': {
                'datatype': 'float',
                'lower_bound': 0.05,
                'upper_bound': 0.95,
                'step': 0.05
            },
            'alpha': {
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
