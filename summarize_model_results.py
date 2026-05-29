import os

import pandas as pd


BASE_PATH = './data/new_data'
OUTPUT_PATH = f'{BASE_PATH}/model_comparison_summary.csv'

RESULT_SOURCES = [
    {
        'model_family': 'logistic_regression',
        'file_path': f'{BASE_PATH}/result_logistic/all_model_results.csv',
    },
    {
        'model_family': 'decision_tree',
        'file_path': f'{BASE_PATH}/result_tree/all_model_results.csv',
    },
    {
        'model_family': 'random_forest',
        'file_path': f'{BASE_PATH}/result_rf/all_model_results.csv',
    },
    {
        'model_family': 'xgboost',
        'file_path': f'{BASE_PATH}/result_xgb/all_model_results.csv',
    },
]


def main():
    summary_frames = []

    for source in RESULT_SOURCES:
        file_path = source['file_path']

        if not os.path.exists(file_path):
            print(f'Skipped missing file: {file_path}')
            continue

        df = pd.read_csv(file_path)
        df.insert(0, 'model_family', source['model_family'])
        summary_frames.append(df)

    if not summary_frames:
        raise FileNotFoundError('No model result files were found.')

    combined_df = pd.concat(summary_frames, ignore_index=True, sort=False)

    preferred_columns = [
        'model_family',
        'model_name',
        'dataset_scope',
        'target_name',
        'use_upsample',
        'cv_accuracy',
        'cv_precision',
        'cv_recall',
        'cv_f1',
        'cv_roc_auc',
        'test_accuracy',
        'test_precision',
        'test_recall',
        'test_f1',
        'test_roc_auc',
    ]

    remaining_columns = [
        column for column in combined_df.columns if column not in preferred_columns
    ]

    combined_df = combined_df[preferred_columns + remaining_columns]
    combined_df = combined_df.sort_values(
        by=['dataset_scope', 'model_family', 'test_roc_auc'],
        ascending=[True, True, False],
    )

    combined_df.to_csv(OUTPUT_PATH, index=False)
    print(combined_df)


if __name__ == '__main__':
    main()
