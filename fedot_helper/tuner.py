from typing import Optional

import optuna

from fedot_helper.data import Data
from fedot_helper.estimator.estimator import Estimator


class Tuner:
    def __init__(self, model: Estimator, scorer: callable, method: str = 'TPE'):
        self.method = method
        self.model = model
        self.scorer = scorer

        self.parameters_tuning_range = self.model.parameters_tuning_range
        optuna.logging.set_verbosity(optuna.logging.INFO)

    def tune(self, x: Data, y: Optional[Data] = None, **kwargs):
        if not self.parameters_tuning_range:
            return self.model, self.scorer(self.model, x, y)

        if self.method in ('TPE', ):
            sampler, pruner = {'TPE': (optuna.samplers.TPESampler(seed=self.model.seed),
                                       optuna.pruners.HyperbandPruner()),
                               }[self.method]

            def objective(trial,
                          model=self.model,
                          params=self.parameters_tuning_range,
                          scorer=self.scorer,
                          x=x,
                          y=y):

                to_set = model.get_params()
                for name, pars in params.items():
                    if pars['type'] is int:
                        to_set[name] = trial.suggest_int(name, pars['min'], pars['max'], pars['step'])
                    elif pars['type'] is float:
                        to_set[name] = trial.suggest_float(name, pars['min'], pars['max'], pars['step'])
                    elif pars['type'] == 'category':
                        to_set[name] = trial.suggest_categorical(name, pars['values'])
                    else:
                        raise ValueError(f"Unknown parameter type {pars['type']}")

                new_model = model.__sklearn_clone__()
                new_model.set_params(**to_set)
                try:
                    new_model.fit(x, y)
                    return scorer(model, x, y)
                except:
                    return scorer(None, x, y)

            study = optuna.create_study(sampler=sampler, pruner=pruner)
            study.optimize(objective, **({'n_jobs': self.model.n_jobs} | kwargs))

            model = self.model.__sklearn_clone__()
            model.set_params(**study.best_params)
            print(model)
            return model, study.best_value
        else:
            raise ValueError(f"Unknown method for tuning `{self.method}`")
