import json
import math
import os

import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier


BASE_PATH = './data/new_data'
RESULT_ROOT = f'{BASE_PATH}/result_xgb'

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

IMPORTANCE_TRIALS = 50
TUNING_TRIALS = 100
CV_SPLITS = 5
CV_TEST_SIZE = 1 / CV_SPLITS
CV_RANDOM_STATE = 20240627
OPTUNA_SEED = 20240505
XGB_RANDOM_STATE = 20240417
N_JOBS_MODEL = -1

os.makedirs(RESULT_ROOT, exist_ok=True)


def get_cv_splitter():
    return StratifiedShuffleSplit(
        n_splits=CV_SPLITS,
        test_size=CV_TEST_SIZE,
        random_state=CV_RANDOM_STATE,
    )


def build_xgb_params(params):
    return {
        'n_estimators': int(params['n_estimators']),
        'max_depth': int(params['max_depth']),
        'learning_rate': float(params['learning_rate']),
        'min_child_weight': float(params['min_child_weight']),
        'subsample': float(params['subsample']),
        'colsample_bytree': float(params['colsample_bytree']),
        'gamma': float(params['gamma']),
        'reg_alpha': float(params['reg_alpha']),
        'reg_lambda': float(params['reg_lambda']),
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'use_label_encoder': False,
        'n_jobs': N_JOBS_MODEL,
        'random_state': XGB_RANDOM_STATE,
    }


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
            sampler = RandomOverSampler(random_state=XGB_RANDOM_STATE)
            X_train, y_train = sampler.fit_resample(X_train, y_train)

        model = XGBClassifier(**build_xgb_params(params))
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


def suggest_xgb_params(trial, search_space):
    if search_space == 'importance':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 10.0),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        }

    return {
        'n_estimators': trial.suggest_int('n_estimators', 150, 400),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.2, log=True),
        'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 8.0),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 3.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 0.5, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 8.0, log=True),
    }


def roc_auc_objective_factory(X, y, use_upsample, search_space):
    def objective(trial):
        params = suggest_xgb_params(trial, search_space)
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
        {'item': 'importance_trials', 'value': IMPORTANCE_TRIALS},
        {'item': 'tuning_trials', 'value': TUNING_TRIALS},
        {'item': 'cv_splits', 'value': CV_SPLITS},
        {'item': 'cv_random_state', 'value': CV_RANDOM_STATE},
        {'item': 'optuna_seed', 'value': OPTUNA_SEED},
        {'item': 'xgb_random_state', 'value': XGB_RANDOM_STATE},
    ]

    pd.DataFrame(config_rows).to_csv(f'{model_dir}/model_config.csv', index=False)


def prepare_candidate_df(candidate_df):
    prepared_df = candidate_df.copy()

    # Prefer a simpler boosting model among candidates within one standard error.
    prepared_df['sort_learning_rate'] = prepared_df['learning_rate']
    prepared_df['sort_reg_alpha'] = -prepared_df['reg_alpha']
    prepared_df['sort_gamma'] = -prepared_df['gamma']
    prepared_df['sort_min_child_weight'] = -prepared_df['min_child_weight']

    return prepared_df


def fit_final_model(X_train, y_train, best_params, use_upsample):
    X_fit = X_train.copy()
    y_fit = y_train.copy()

    if use_upsample:
        sampler = RandomOverSampler(random_state=XGB_RANDOM_STATE)
        X_fit, y_fit = sampler.fit_resample(X_fit, y_fit)

    model = XGBClassifier(**build_xgb_params(best_params))
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

    importance_study = optuna.create_study(
        directions=['maximize'],
        sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED),
    )
    importance_study.optimize(
        roc_auc_objective_factory(
            X=X_train,
            y=y_train,
            use_upsample=config['use_upsample'],
            search_space='importance',
        ),
        n_trials=IMPORTANCE_TRIALS,
        n_jobs=-1,
    )

    param_importances = optuna.importance.get_param_importances(importance_study)
    importance_df = pd.DataFrame({
        'parameter': list(param_importances.keys()),
        'importance': list(param_importances.values()),
    })
    importance_df.to_csv(f'{model_dir}/hyperparameter_importance.csv', index=False)

    tuning_study = optuna.create_study(
        directions=['maximize'],
        sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED),
    )
    tuning_study.optimize(
        roc_auc_objective_factory(
            X=X_train,
            y=y_train,
            use_upsample=config['use_upsample'],
            search_space='tuning',
        ),
        n_trials=TUNING_TRIALS,
        n_jobs=-1,
    )

    tuning_trials_df = pd.DataFrame([trial_to_record(trial) for trial in tuning_study.trials])
    tuning_trials_df.to_csv(f'{model_dir}/tuning_trials.csv', index=False)
    save_trials_jsonl(f'{model_dir}/tuning_trials.jsonl', tuning_study.trials)
    save_json(f'{model_dir}/trial_best.json', trial_to_record(tuning_study.best_trial))

    within_one_threshold = tuning_study.best_trial.value - tuning_study.best_trial.user_attrs['std_err']
    candidate_df = tuning_trials_df.loc[tuning_trials_df['value'] >= within_one_threshold].copy()
    candidate_df = prepare_candidate_df(candidate_df)

    best_within_one_row = candidate_df.sort_values(
        [
            'sort_learning_rate',
            'sort_reg_alpha',
            'sort_gamma',
            'sort_min_child_weight',
            'max_depth',
            'n_estimators',
        ]
    ).iloc[0].to_dict()

    pd.DataFrame([best_within_one_row]).to_csv(f'{model_dir}/best_within_one.csv', index=False)
    save_json(f'{model_dir}/best_within_one.json', best_within_one_row)

    best_params = {
        'n_estimators': best_within_one_row['n_estimators'],
        'max_depth': best_within_one_row['max_depth'],
        'learning_rate': best_within_one_row['learning_rate'],
        'min_child_weight': best_within_one_row['min_child_weight'],
        'subsample': best_within_one_row['subsample'],
        'colsample_bytree': best_within_one_row['colsample_bytree'],
        'gamma': best_within_one_row['gamma'],
        'reg_alpha': best_within_one_row['reg_alpha'],
        'reg_lambda': best_within_one_row['reg_lambda'],
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

    feature_importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': final_model.feature_importances_,
    }).sort_values(by='importance', ascending=False)

    final_params_df = pd.DataFrame({
        'parameter': list(build_xgb_params(best_params).keys()),
        'value': list(build_xgb_params(best_params).values()),
    })

    cv_fold_df.to_csv(f'{model_dir}/cv_fold_results.csv', index=False)
    cv_summary_df.to_csv(f'{model_dir}/cv_summary.csv', index=False)
    test_metrics_df.to_csv(f'{model_dir}/test_metrics.csv', index=False)
    test_predictions_df.to_csv(f'{model_dir}/test_predictions.csv', index=False)
    feature_importance_df.to_csv(f'{model_dir}/feature_importances.csv', index=False)
    final_params_df.to_csv(f'{model_dir}/final_xgb_params.csv', index=False)

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
        'selected_n_estimators': build_xgb_params(best_params)['n_estimators'],
        'selected_max_depth': build_xgb_params(best_params)['max_depth'],
        'selected_learning_rate': build_xgb_params(best_params)['learning_rate'],
        'selected_min_child_weight': build_xgb_params(best_params)['min_child_weight'],
        'selected_subsample': build_xgb_params(best_params)['subsample'],
        'selected_colsample_bytree': build_xgb_params(best_params)['colsample_bytree'],
        'selected_gamma': build_xgb_params(best_params)['gamma'],
        'selected_reg_alpha': build_xgb_params(best_params)['reg_alpha'],
        'selected_reg_lambda': build_xgb_params(best_params)['reg_lambda'],
    }


summary_rows = []
for config in MODEL_CONFIGS:
    summary_rows.append(run_single_model(config))

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f'{RESULT_ROOT}/all_model_results.csv', index=False)

print(summary_df)
