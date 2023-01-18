import sklearn

from . import _sklearn_model


class RandomForest(_sklearn_model.SklearnModel):
    """
    Implementation of a class for Random Forest.

    See :obj:`~thermpred.model._base_model.BaseModel` for more information on the attributes.
    """
    featureset = 'features'

    def define_model(self):
        """
        Definition of the actual prediction model.

        See :obj:`~thermpred.model._base_model.BaseModel` for more information.
        """
        # all hyperparameters defined are suggested for optimization
        params = self.suggest_all_hyperparams_to_optuna()
        # add random_state for reproducibility and n_jobs for multiprocessing
        params.update({'random_state': 42, 'n_jobs': -1})
        if self.task == 'classification':
            return sklearn.ensemble.RandomForestClassifier(**params)
        else:
            return sklearn.ensemble.RandomForestRegressor(**params)

    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~thermpred.model._base_model.BaseModel` for more information on the format.
        """
        return {
            'n_estimators': {
                'datatype': 'categorical',
                'list_of_values': [50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000,
                                   3500, 4000, 4500, 5000]
            },
            'min_samples_split': {
                'datatype': 'float',
                'lower_bound': 0.005,
                'upper_bound': 0.9,
                'step': 0.005
            },
            'max_depth': {
                'datatype': 'int',
                'lower_bound': 2,
                'upper_bound': 50,
                'step': 2
            },
            'min_samples_leaf': {
                'datatype': 'float',
                'lower_bound': 0.005,
                'upper_bound': 0.5,
                'step': 0.005
            },
            'max_features': {
                'datatype': 'categorical',
                'list_of_values': ['sqrt', 'log2']
            }
        }
