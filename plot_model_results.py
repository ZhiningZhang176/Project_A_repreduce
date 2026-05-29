import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_PATH = './data/new_data'
FIGURE_DIR = f'{BASE_PATH}/figures'
COMPARISON_OUTPUT = f'{FIGURE_DIR}/model_performance_comparison.png'
FACE_MASK_OUTPUT = f'{FIGURE_DIR}/face_mask_top_predictors.png'
PROTECTIVE_OUTPUT = f'{FIGURE_DIR}/protective_behaviour_top_predictors.png'

MODEL_RESULT_FILES = {
    'Logistic Regression': f'{BASE_PATH}/result_logistic/all_model_results.csv',
    'Decision Tree': f'{BASE_PATH}/result_tree/all_model_results.csv',
    'Random Forest': f'{BASE_PATH}/result_rf/all_model_results.csv',
    'XGBoost': f'{BASE_PATH}/result_xgb/all_model_results.csv',
}

FEATURE_IMPORTANCE_SOURCES = {
    'face_mask': [
        {
            'label': 'All-time',
            'path': f'{BASE_PATH}/result_xgb/model_1_all_time_face_mask/feature_importances.csv',
        },
        {
            'label': 'Non-mandate',
            'path': f'{BASE_PATH}/result_xgb/model_1a_non_mandate_face_mask/feature_importances.csv',
        },
        {
            'label': 'Mandate',
            'path': f'{BASE_PATH}/result_xgb/model_1b_mandate_face_mask/feature_importances.csv',
        },
    ],
    'protective': [
        {
            'label': 'All-time',
            'path': f'{BASE_PATH}/result_xgb/model_2_all_time_protective_behaviour/feature_importances.csv',
        },
        {
            'label': 'Non-mandate',
            'path': f'{BASE_PATH}/result_xgb/model_2a_non_mandate_protective_behaviour/feature_importances.csv',
        },
        {
            'label': 'Mandate',
            'path': f'{BASE_PATH}/result_xgb/model_2b_mandate_protective_behaviour/feature_importances.csv',
        },
    ],
}

DATASET_ORDER = [
    'all_time_face_mask',
    'non_mandate_face_mask',
    'mandate_face_mask',
    'all_time_protective_behaviour',
    'non_mandate_protective_behaviour',
    'mandate_protective_behaviour',
]

DATASET_LABELS = {
    'all_time_face_mask': 'Face mask\nAll-time',
    'non_mandate_face_mask': 'Face mask\nNon-mandate',
    'mandate_face_mask': 'Face mask\nMandate',
    'all_time_protective_behaviour': 'Protective\nAll-time',
    'non_mandate_protective_behaviour': 'Protective\nNon-mandate',
    'mandate_protective_behaviour': 'Protective\nMandate',
}


def ensure_output_dir():
    os.makedirs(FIGURE_DIR, exist_ok=True)


def load_model_results():
    frames = []

    for model_family, file_path in MODEL_RESULT_FILES.items():
        if not os.path.exists(file_path):
            print(f'Skipped missing file: {file_path}')
            continue

        df = pd.read_csv(file_path)
        df['model_family'] = model_family
        frames.append(df)

    if not frames:
        raise FileNotFoundError('No model result files were found.')

    combined_df = pd.concat(frames, ignore_index=True)
    combined_df = combined_df[combined_df['dataset_scope'].isin(DATASET_ORDER)].copy()
    combined_df['dataset_scope'] = pd.Categorical(
        combined_df['dataset_scope'],
        categories=DATASET_ORDER,
        ordered=True,
    )
    return combined_df.sort_values(['dataset_scope', 'model_family'])


def plot_model_performance(combined_df):
    plot_df = combined_df.copy()
    plot_df['dataset_label'] = plot_df['dataset_scope'].map(DATASET_LABELS)

    plt.figure(figsize=(13, 6.5))
    sns.barplot(
        data=plot_df,
        x='dataset_label',
        y='test_roc_auc',
        hue='model_family',
        palette='Set2',
    )
    plt.xlabel('')
    plt.ylabel('Test ROC-AUC')
    plt.title('Model Performance Across the Six Datasets')
    plt.ylim(0.65, 0.95)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(COMPARISON_OUTPUT, dpi=300, bbox_inches='tight')
    plt.close()


def load_top_features(group_name, top_n=10):
    frames = []

    for source in FEATURE_IMPORTANCE_SOURCES[group_name]:
        if not os.path.exists(source['path']):
            print(f'Skipped missing file: {source["path"]}')
            continue

        df = pd.read_csv(source['path'])
        df = df.sort_values('importance', ascending=False).head(top_n).copy()
        df['period'] = source['label']
        frames.append(df[['feature', 'importance', 'period']])

    if not frames:
        raise FileNotFoundError(f'No feature importance files were found for {group_name}.')

    combined_df = pd.concat(frames, ignore_index=True)

    feature_order = (
        combined_df.groupby('feature')['importance']
        .max()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    combined_df['feature'] = pd.Categorical(
        combined_df['feature'],
        categories=feature_order[::-1],
        ordered=True,
    )

    return combined_df.sort_values('feature')


def plot_top_features(group_name, output_path, title):
    feature_df = load_top_features(group_name)

    plt.figure(figsize=(11, 8))
    sns.barplot(
        data=feature_df,
        x='importance',
        y='feature',
        hue='period',
        palette='Set2',
        orient='h',
    )
    plt.xlabel('Feature Importance')
    plt.ylabel('')
    plt.title(title)
    plt.legend(title='Period')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    sns.set_theme(style='whitegrid', context='talk')
    ensure_output_dir()

    combined_df = load_model_results()
    plot_model_performance(combined_df)

    plot_top_features(
        group_name='face_mask',
        output_path=FACE_MASK_OUTPUT,
        title='Top XGBoost Predictors of Face Mask Behaviour',
    )
    plot_top_features(
        group_name='protective',
        output_path=PROTECTIVE_OUTPUT,
        title='Top XGBoost Predictors of Protective Behaviour',
    )

    print('Saved figures:')
    print(COMPARISON_OUTPUT)
    print(FACE_MASK_OUTPUT)
    print(PROTECTIVE_OUTPUT)


if __name__ == '__main__':
    main()
