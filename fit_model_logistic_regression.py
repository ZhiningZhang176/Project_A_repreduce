import json
import math
import os

import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


BASE_PATH = './data/new_data'
RESULT_ROOT = f'{BASE_PATH}/result_logistic'

MODEL_CONFIGS = [
    {
        'model_name': 'model_1',
        'dataset_scope': 'all_time_face_mask',
        'target_name': 'face_mask_behaviour_binary',
        'use_upsample': False,
    },
    {
        'model_name': 'model_2',
        'dataset_scope': 'all_time_protective_behaviour',
        'target_name': 'protective_behaviour_binary',
        'use_upsample': False,
    },
    {
        'model_name': 'model_1a',
        'dataset_scope': 'non_mandate_face_mask',
        'target_name': 'face_mask_behaviour_binary',
        'use_upsample': True,
    },
    {
        'model_name': 'model_2a',
        'dataset_scope': 'non_mandate_protective_behaviour',
        'target_name': 'protective_behaviour_binary',
        'use_upsample': True,
    },
    {
        'model_name': 'model_1b',
        'dataset_scope': 'mandate_face_mask',
        'target_name': 'face_mask_behaviour_binary',
        'use_upsample': True,
    },
    {
        'model_name': 'model_2b',
        'dataset_scope': 'mandate_protective_behaviour',
        'target_name': 'protective_behaviour_binary',
        'use_upsample': True,
    },
]

TUNING_TRIALS = 100
CV_SPLITS = 5
CV_TEST_SIZE = 1 / CV_SPLITS
CV_RANDOM_STATE = 20240627
OPTUNA_SEED = 20240505
LOGISTIC_RANDOM_STATE = 20240417
MAX_ITER = 5000

os.makedirs(RESULT_ROOT, exist_ok=True)


def get_cv_splitter():
    return StratifiedShuffleSplit(
        n_splits=CV_SPLITS,
        test_size=CV_TEST_SIZE,
        random_state=CV_RANDOM_STATE,
    )


def normalize_class_weight(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return value


def build_logistic_params(params):
    logistic_params = {
        'C': float(params['C']),
        'penalty': params['penalty'],
        'solver': params['solver'],
        'fit_intercept': bool(params['fit_intercept']),
        'class_weight': normalize_class_weight(params.get('class_weight')),
        'max_iter': MAX_ITER,
        'random_state': LOGISTIC_RANDOM_STATE,
    }
    return logistic_params


def evaluate_metrics(y_true, y_pred, y_prob):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
    }


def compute_cv_scores(X, y, params, use_upsample):
    splitter = get_cv_splitter()
    fold_results = []

    for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        X_train = X.iloc[train_idx].copy()
        y_train = y.iloc[train_idx].copy()
        X_valid = X.iloc[valid_idx].copy()
        y_valid = y.iloc[valid_idx].copy()

        if use_upsample:
            sampler = RandomOverSampler()
            X_train, y_train = sampler.fit_resample(X_train, y_train)

        model = LogisticRegression(**build_logistic_params(params))
        model.fit(X_train, y_train)

        y_pred = model.predict(X_valid)
        y_prob = model.predict_proba(X_valid)[:, 1]

        metrics = evaluate_metrics(y_valid, y_pred, y_prob)
        metrics['fold'] = fold_index
        fold_results.append(metrics)

    return pd.DataFrame(fold_results)[
        ['fold', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    ]


def cv_summary_from_folds(cv_fold_df):
    return pd.DataFrame({
        'metric': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        'mean_score': [
            cv_fold_df['accuracy'].mean(),
            cv_fold_df['precision'].mean(),
            cv_fold_df['recall'].mean(),
            cv_fold_df['f1'].mean(),
            cv_fold_df['roc_auc'].mean(),
        ],
        'std_score': [
            cv_fold_df['accuracy'].std(),
            cv_fold_df['precision'].std(),
            cv_fold_df['recall'].std(),
            cv_fold_df['f1'].std(),
            cv_fold_df['roc_auc'].std(),
        ],
    })


def suggest_logistic_params(trial):
    penalty_solver = trial.suggest_categorical(
        'penalty_solver',
        [
            'l1__liblinear',
            'l1__saga',
            'l2__lbfgs',
            'l2__liblinear',
            'l2__saga',
        ],
    )
    penalty, solver = penalty_solver.split('__')
    trial.set_user_attr('penalty', penalty)
    trial.set_user_attr('solver', solver)

    return {
        'C': trial.suggest_float('C', 1e-3, 50.0, log=True),
        'penalty': penalty,
        'solver': solver,
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
    }


def roc_auc_objective_factory(X, y, use_upsample):
    def objective(trial):
        params = suggest_logistic_params(trial)
        cv_fold_df = compute_cv_scores(
            X=X,
            y=y,
            params=params,
            use_upsample=use_upsample,
        )

        mean_roc_auc = cv_fold_df['roc_auc'].mean()
        std_err = cv_fold_df['roc_auc'].std(ddof=0) / math.sqrt(len(cv_fold_df))

        trial.set_user_attr('std_err', float(std_err))
        return float(mean_roc_auc)

    return objective


def trial_to_record(trial):
    record = {
        'number': trial.number,
        'value': trial.value,
    }

    for key, value in trial.params.items():
        record[key] = value

    for key, value in trial.user_attrs.items():
        record[key] = value

    return record


def save_json(path, payload):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, default=str)


def save_trials_jsonl(path, trials):
    with open(path, 'w', encoding='utf-8') as file:
        for trial in trials:
            file.write(json.dumps(trial_to_record(trial), default=str))
            file.write('\n')


def save_model_config(config, model_dir):
    config_rows = [
        {'item': 'model_name', 'value': config['model_name']},
        {'item': 'dataset_scope', 'value': config['dataset_scope']},
        {'item': 'target_name', 'value': config['target_name']},
        {'item': 'use_upsample', 'value': config['use_upsample']},
        {'item': 'train_feature_file', 'value': f"{BASE_PATH}/X_train_{config['model_name']}.csv"},
        {'item': 'test_feature_file', 'value': f"{BASE_PATH}/X_test_{config['model_name']}.csv"},
        {'item': 'train_label_file', 'value': f"{BASE_PATH}/y_train_{config['model_name']}.csv"},
        {'item': 'test_label_file', 'value': f"{BASE_PATH}/y_test_{config['model_name']}.csv"},
        {'item': 'tuning_trials', 'value': TUNING_TRIALS},
        {'item': 'cv_splits', 'value': CV_SPLITS},
        {'item': 'cv_random_state', 'value': CV_RANDOM_STATE},
        {'item': 'optuna_seed', 'value': OPTUNA_SEED},
        {'item': 'max_iter', 'value': MAX_ITER},
    ]

    pd.DataFrame(config_rows).to_csv(f'{model_dir}/model_config.csv', index=False)


def prepare_candidate_df(candidate_df):
    prepared_df = candidate_df.copy()
    prepared_df['class_weight_key'] = prepared_df['class_weight'].astype(str).replace({'nan': 'None'})
    prepared_df['penalty_rank'] = prepared_df['penalty'].map({'l1': 0, 'l2': 1})
    prepared_df['class_weight_rank'] = prepared_df['class_weight_key'].map({'None': 0, 'balanced': 1})
    prepared_df['solver_rank'] = prepared_df['solver'].map({'liblinear': 0, 'lbfgs': 1, 'saga': 2})
    return prepared_df


def fit_final_model(X_train, y_train, best_params, use_upsample):
    X_fit = X_train.copy()
    y_fit = y_train.copy()

    if use_upsample:
        sampler = RandomOverSampler()
        X_fit, y_fit = sampler.fit_resample(X_fit, y_fit)

    model = LogisticRegression(**build_logistic_params(best_params))
    model.fit(X_fit, y_fit)
    return model


def run_single_model(config):
    model_name = config['model_name']
    model_dir = f"{RESULT_ROOT}/{model_name}_{config['dataset_scope']}"
    os.makedirs(model_dir, exist_ok=True)

    X_train = pd.read_csv(f'{BASE_PATH}/X_train_{model_name}.csv')
    X_test = pd.read_csv(f'{BASE_PATH}/X_test_{model_name}.csv')
    y_train = pd.read_csv(f'{BASE_PATH}/y_train_{model_name}.csv')['y_train']
    y_test = pd.read_csv(f'{BASE_PATH}/y_test_{model_name}.csv')['y_test']

    save_model_config(config, model_dir)

    tuning_study = optuna.create_study(
        directions=['maximize'],
        sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED),
    )
    tuning_study.optimize(
        roc_auc_objective_factory(
            X=X_train,
            y=y_train,
            use_upsample=config['use_upsample'],
        ),
        n_trials=TUNING_TRIALS,
        n_jobs=-1,
    )

    tuning_trials_df = pd.DataFrame([trial_to_record(trial) for trial in tuning_study.trials])
    tuning_trials_df.to_csv(f'{model_dir}/tuning_trials.csv', index=False)
    save_trials_jsonl(f'{model_dir}/tuning_trials.jsonl', tuning_study.trials)
    save_json(f'{model_dir}/trial_best.json', trial_to_record(tuning_study.best_trial))

    within_one_threshold = tuning_study.best_trial.value - tuning_study.best_trial.user_attrs['std_err']
    candidate_df = tuning_trials_df.loc[tuning_trials_df['value'] > within_one_threshold].copy()
    candidate_df = prepare_candidate_df(candidate_df)

    best_within_one_row = candidate_df.sort_values(
        ['C', 'penalty_rank', 'class_weight_rank', 'solver_rank']
    ).iloc[0].to_dict()

    pd.DataFrame([best_within_one_row]).to_csv(f'{model_dir}/best_within_one.csv', index=False)
    save_json(f'{model_dir}/best_within_one.json', best_within_one_row)

    best_params = {
        'C': best_within_one_row['C'],
        'penalty': best_within_one_row['penalty'],
        'solver': best_within_one_row['solver'],
        'fit_intercept': best_within_one_row['fit_intercept'],
        'class_weight': best_within_one_row['class_weight'],
    }

    cv_fold_df = compute_cv_scores(
        X=X_train,
        y=y_train,
        params=best_params,
        use_upsample=config['use_upsample'],
    )
    cv_summary_df = cv_summary_from_folds(cv_fold_df)

    final_model = fit_final_model(
        X_train=X_train,
        y_train=y_train,
        best_params=best_params,
        use_upsample=config['use_upsample'],
    )

    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_metrics(y_test, y_pred, y_prob)

    test_metrics_df = pd.DataFrame({
        'metric': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        'score': [
            test_metrics['accuracy'],
            test_metrics['precision'],
            test_metrics['recall'],
            test_metrics['f1'],
            test_metrics['roc_auc'],
        ],
    })

    test_predictions_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
    })

    coefficient_df = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': final_model.coef_[0],
    }).sort_values(by='coefficient', ascending=False)

    final_params_df = pd.DataFrame({
        'parameter': list(build_logistic_params(best_params).keys()),
        'value': list(build_logistic_params(best_params).values()),
    })

    cv_fold_df.to_csv(f'{model_dir}/cv_fold_results.csv', index=False)
    cv_summary_df.to_csv(f'{model_dir}/cv_summary.csv', index=False)
    test_metrics_df.to_csv(f'{model_dir}/test_metrics.csv', index=False)
    test_predictions_df.to_csv(f'{model_dir}/test_predictions.csv', index=False)
    coefficient_df.to_csv(f'{model_dir}/coefficients.csv', index=False)
    final_params_df.to_csv(f'{model_dir}/final_logistic_params.csv', index=False)

    return {
        'model_name': model_name,
        'dataset_scope': config['dataset_scope'],
        'target_name': config['target_name'],
        'use_upsample': config['use_upsample'],
        'cv_accuracy': cv_summary_df.loc[cv_summary_df['metric'] == 'accuracy', 'mean_score'].iloc[0],
        'cv_precision': cv_summary_df.loc[cv_summary_df['metric'] == 'precision', 'mean_score'].iloc[0],
        'cv_recall': cv_summary_df.loc[cv_summary_df['metric'] == 'recall', 'mean_score'].iloc[0],
        'cv_f1': cv_summary_df.loc[cv_summary_df['metric'] == 'f1', 'mean_score'].iloc[0],
        'cv_roc_auc': cv_summary_df.loc[cv_summary_df['metric'] == 'roc_auc', 'mean_score'].iloc[0],
        'test_accuracy': test_metrics['accuracy'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f1'],
        'test_roc_auc': test_metrics['roc_auc'],
        'selected_C': build_logistic_params(best_params)['C'],
        'selected_penalty': build_logistic_params(best_params)['penalty'],
        'selected_solver': build_logistic_params(best_params)['solver'],
        'selected_fit_intercept': build_logistic_params(best_params)['fit_intercept'],
        'selected_class_weight': build_logistic_params(best_params)['class_weight'],
    }


summary_rows = []
for config in MODEL_CONFIGS:
    summary_rows.append(run_single_model(config))

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f'{RESULT_ROOT}/all_model_results.csv', index=False)

print(summary_df)
