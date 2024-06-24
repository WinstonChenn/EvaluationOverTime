"""Evaluator class. 

For a given model class (implementing the ExpModel interface) and
an EotDataset object, the Evaluator evaluates both in-sample and 
out-of sample test performance.
"""

import pickle
import os
import numpy as np

from emdot.eot_dataset import EotDataset
from emdot.models.ExpModel import ExpModel
from tqdm import tqdm
from emdot.utils import dict_combos
from typing import Optional, Callable
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

class EotEvaluator:
    def __init__(
        self, 
        dataset: EotDataset, 
        method: str, 
        model: ExpModel, 
        model_name: str, 
        search_space: dict, 
        eval_func: Callable
        ):
        """Initializes Evaluator object.

        Args:
            dataset: EotDataset object
            method: method name (e.g. "all_historical", "sliding_window")
            model: ExpModel object (e.g. LR, GBDT, MLP)
            model_name: name of the ML models (e.g. "LR", "GBDT", "MLP")
            search_space: dictionary of potential hyperparameters to search for best model
                e.g. for GBDT, {
                        "n_estimators": [50, 100],
                        "max_depth": [3, 5],
                        "learning_rate": [0.01, 0.1]
                    }
        """
        # assert issubclass(model, ExpModel)
        
        self.dataset = dataset
        self.method = method
        self.model = model
        self.model_name = model_name
        self.search_space = search_space
        self.eval_func = eval_func
        
        self.best_model = None
        self.best_hparams = None

    def load_best_model(self, model_folder):
        path = self.get_model_path(model_folder)
        assert os.path.isfile(path)
        with open(path, "rb") as f:
            self.best_model = pickle.load(f)
        f.close()
        self.best_hparams = self.best_model.get_params()
        
            
    def fit_best_model(self, eval_metric: str = "roc_auc", n_iter: int = 100, n_jobs: int = 100, 
                       verbose: int=1, n_fold: int=5):
        """Performs grid search over the self.search_space hyperparameter grid.
        
        Selects the trained model with the best validation performance.
        Stores the best results, model, and hyperparameters as attributes of Evaluator.

        Args:
            eval_metric: name of metric to select best model on. This metric should be
                returned in a dictionary provided by the evaluate() function in the 
                ExpModel in self.model. Assumes that higher is better.
        """

        ## Get_dataset (no explicite validation set, use CV)
        X_train, y_train = self.dataset.get_train(return_df=False)
        X_val, y_val = self.dataset.get_val(return_df=False)
        X_train = np.vstack((X_train, X_val))
        y_train = np.concatenate((y_train, y_val))

        clf = RandomizedSearchCV(self.model(), self.search_space, n_iter=n_iter, n_jobs=n_jobs, 
                                 verbose=verbose, cv=StratifiedKFold(n_fold), scoring=eval_metric)
        clf.fit(X_train, y_train)
        self.best_model = clf.best_estimator_
        self.best_hparams = clf.best_params_
                
    def in_sample_test(self):
        """Evaluates in-sample testing performance.

        Returns:
            dictionary containing metadata about the evaluation and results
        """
        assert self.best_model is not None, "Forget to fit the model"
        
        ## Get in-sample test data
        X_test, y_test = self.dataset.get_test_in_sample(return_df=False)
        in_sample_result = self.eval_func(self.best_model, X_test, y_test)
        
        return {
            "model": self.model_name,
            "test_type": "insample",
            "train_start": self.dataset.train_start_t,
            "train_end": self.dataset.train_end_t,
            "test_start": self.dataset.train_start_t,
            "test_end": self.dataset.train_end_t,
            **in_sample_result,
            **self.dataset.get_dataset_info(),
            "test_size": X_test.shape[0],
            "best_hparams": self.best_hparams
        }
        
    def out_sample_test(
        self, 
        test_start_year: int, 
        test_end_year: int = None
        ):
        """Evaluate the out-of-sample testing performance.

        Args:
            test_start_year: starting time point of out-sample test
            test_end_year: end time point of out-sample test
                If None, set to the same as test_start_year
        
        Returns:
            dictionary containing metadata about the evaluation and results
        
        """
        assert self.best_model is not None, "Please run fit_best_model first."
        
        if test_end_year is None:
            test_end_year = test_start_year
        
        ## Get out-sample test data
        X_test, y_test = self.dataset.get_test_out_sample(test_start_year, 
            test_end_year=test_end_year, return_df=False)
        ## Evaluate out-sample test performance
        out_sample_result = self.eval_func(self.best_model, X_test, y_test)
        
        return {
            "model": self.model_name,
            "test_type": "outsample",
            "train_start": self.dataset.train_start_t,
            "train_end": self.dataset.train_end_t,
            "test_start": test_start_year,
            "test_end": test_end_year,
            **out_sample_result,
            **self.dataset.get_dataset_info(),
            "test_size": X_test.shape[0],
            "best_hparams": self.best_hparams
        }
        
    def save_best_model(self, model_folder="./model_info"):
        """Store trained models and save coefficients of each feature for the model if it has coefficients.
        
        Args:
            model_folder: folder to store models

        Returns:
            dictionary of coefficients if the model has a coef_ attribute. Otherwise, None.
        """

        if model_folder is not None:
            ## Store the model
            model_path = self.get_model_path(model_folder=model_folder)
            with open(model_path, "wb") as f:
                pickle.dump(self.best_model, f)
            f.close()
        
    def get_model_path(self, model_folder="./model_info"):
        return os.path.join(model_folder, f"{str(self.dataset.train_start_t)}_{str(self.dataset.train_end_t)}_{self.method}.pkl")

    def get_coef_dict(self):
        # For models where get_coefs() does not return None, 
        # return a dictionary containing the coefficients and some experiment parameters
        feature_names = self.dataset.get_feature_list()
        coefs = self.best_model.get_coefs(feature_names)
        if coefs is not None:
            dict_coef = {
                "train_start": self.dataset.train_start_t,
                "train_end": self.dataset.train_end_t,
                "seed": self.dataset.seed,
                **coefs
            }        
            return dict_coef
        return None
