import re
import pandas as pd
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from lightgbm import log_evaluation, early_stopping

from helpers import get_settings
config = get_settings()



class EssayScorerClassic:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.models_with_tfidf = None

    def create_vectorizer(self):
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            token_pattern=None,
            strip_accents='unicode',
            analyzer='word',
            ngram_range=(1, 4),
            min_df=0.05,
            max_df=0.95,
            sublinear_tf=True,
        )
        self.column_transformer = ColumnTransformer(
            [("vectorizer", self.vectorizer, 'full_text')],
            remainder='passthrough'
        )

    def create_model(self):
        self.model = LGBMRegressor(
            objective=self.qwk_obj,
            metrics='None',
            learning_rate=0.1,
            max_depth=5,
            num_leaves=10,
            colsample_bytree=0.5,
            reg_alpha=0.1,
            reg_lambda=0.8,
            n_estimators=1024,
            random_state=42,
            extra_trees=True,
            class_weight='balanced',
            verbosity=-1
        )

    def qwk_obj(self, y_true, y_pred):
        a = config.QWK_A
        b = config.QWK_B
        labels = y_true + a
        preds = y_pred + a
        preds = preds.clip(1, 6)
        f = 1/2*np.sum((preds-labels)**2)
        g = 1/2*np.sum((preds-a)**2+b)
        df = preds - labels
        dg = preds - a
        grad = (df/g - f*dg/g**2)*len(labels)
        hess = np.ones(len(labels))
        return grad, hess

    def quadratic_weighted_kappa(self, y_true, y_pred):
        a = config.QWK_A
        y_true = y_true + a
        y_pred = (y_pred + a).clip(1, 6).round()
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        return 'QWK', qwk, True
    
    def prepare_y(self,y_raw):

        # Converting the 'score' column to integer type and assigning to y
        y_split = y_raw.astype(int).values
        y = y_raw.astype(np.float32).values - config.QWK_A
        return y, y_split

    def train_model(self, X, y):

        # prepare y
        y, y_split =self.prepare_y(y)

        skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=0)
        f1_scores = []
        kappa_scores = []
        models_with_tfidf = []
        predictions = []
        callbacks = [log_evaluation(period=25), early_stopping(stopping_rounds=75, first_metric_only=True)]

        for i, (train_index, test_index) in enumerate(skf.split(X, y_split), 1):
            print(f'fold {i}')
            X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
            y_train_fold, y_test_fold, y_test_fold_int = y[train_index], y[test_index], y_split[test_index]

            X_train_fold = self.column_transformer.fit_transform(X_train_fold)
            X_test_fold = self.column_transformer.transform(X_test_fold)

            predictor = self.model.fit(
                X_train_fold, y_train_fold,
                eval_names=['train', 'valid'],
                eval_set=[(X_train_fold, y_train_fold), (X_test_fold, y_test_fold)],
                eval_metric=self.quadratic_weighted_kappa,
                callbacks=callbacks
            )

            models_with_tfidf.append((self.column_transformer, predictor))

            predictions_fold = predictor.predict(X_test_fold) + config.QWK_A
            predictions_fold = predictions_fold.clip(1, 6).round()
            predictions.append(predictions_fold)

            f1_fold = f1_score(y_test_fold_int, predictions_fold, average='weighted')
            f1_scores.append(f1_fold)

            kappa_fold = cohen_kappa_score(y_test_fold_int, predictions_fold, weights='quadratic')
            kappa_scores.append(kappa_fold)

            cm = confusion_matrix(y_test_fold_int, predictions_fold, labels=[x for x in range(1, 7)])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[x for x in range(1, 7)])
          #  disp.plot()
            plt.show()
            print(f'F1 score across fold: {f1_fold}')
            print(f'Cohen kappa score across fold: {kappa_fold}')

        mean_f1_score = np.mean(f1_scores)
        mean_kappa_score = np.mean(kappa_scores)

        print(f'Mean F1 score across 15 folds: {mean_f1_score}')
        print(f'Mean Cohen kappa score across 15 folds: {mean_kappa_score}')

        self.models_with_tfidf = models_with_tfidf

        return models_with_tfidf

    def predict(self, test_feats):
        probabilities = []

        if self.models_with_tfidf == None:
            print('no fitted models yet')
            return None

        for pipe in self.models_with_tfidf:
            test_transformed = pipe[0].transform(test_feats)
            proba = pipe[1].predict(test_transformed) + config.QWK_A
            probabilities.append(proba)

        predictions = np.mean(probabilities, axis=0)
        predictions = np.round(predictions.clip(1, 6))
        return predictions
