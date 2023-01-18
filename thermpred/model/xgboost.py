import xgboost

from . import _sklearn_model


class XgBoost(_sklearn_model.SklearnModel):
    """
    Implementation of a class for XGBoost.

    See :obj:`~thermpred.model._base_model.BaseModel` for more information on the attributes.
    """
    featureset = 'features'

    def define_model(self) -> xgboost.XGBModel:
        """
        Definition of the actual prediction model.

        See :obj:`~thermpred.model._base_model.BaseModel` for more information.
        """
        # all hyperparameters defined for XGBoost are suggested for optimization
        params = self.suggest_all_hyperparams_to_optuna()
        # add random_state for reproducibility
        params.update({'random_state': 42, 'reg_lambda': 0})
        if self.task == 'classification':
            # set some parameters to prevent warnings
            params.update({'use_label_encoder': False})
            eval_metric = 'mlogloss' if self.n_outputs > 2 else 'logloss'
            params.update({'eval_metric': eval_metric})
            return xgboost.XGBClassifier(**params)
        else:
            return xgboost.XGBRegressor(**params)

    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~thermpred.model._base_model.BaseModel` for more information on the format.

        Further params that potentially can be optimized

            .. code-block:: python

                'reg_lambda': {
                    'datatype': 'float',
                    'lower_bound': 0,
                    'upper_bound': 1000,
                    'step': 10
                }

        """
        return {
            'n_estimators': {
                'datatype': 'categorical',
                'list_of_values': [50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]
            },
            'learning_rate': {
                    'datatype': 'float',
                    'lower_bound': 0.025,
                    'upper_bound': 0.5,
                    'step': 0.025
            },
            'max_depth': {
                'datatype': 'int',
                'lower_bound': 2,
                'upper_bound': 20
            },
            'gamma': {
                'datatype': 'float',
                'lower_bound': 0,
                'upper_bound': 10,
                'step': 0.1
            },
            'subsample': {
                'datatype': 'float',
                'lower_bound': 0.05,
                'upper_bound': 0.95,
                'step': 0.05
            },
            'colsample_bytree': {
                'datatype': 'float',
                'lower_bound': 0.05,
                'upper_bound': 0.95,
                'step': 0.05
            },
            'reg_alpha': {
                'datatype': 'float',
                'lower_bound': 0,
                'upper_bound': 10,
                'step': 0.1
            }
        }
