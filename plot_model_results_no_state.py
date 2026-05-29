import os

import matplotlib.pyplot as plt
import pandas as pd


BASE_PATH = './data/new_data'
FIGURE_DIR = f'{BASE_PATH}/figures'

FACE_MASK_OUTPUT = f'{FIGURE_DIR}/face_mask_top_predictors_no_state.png'
PROTECTIVE_OUTPUT = f'{FIGURE_DIR}/protective_behaviour_top_predictors_no_state.png'

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


def ensure_output_dir():
    os.makedirs(FIGURE_DIR, exist_ok=True)


PHQ_LABELS = {
    'PHQ4_1': 'Little interest or pleasure in doing things',
    'PHQ4_2': 'Feeling down, depressed, or hopeless',
    'PHQ4_3': 'Feeling nervous, anxious, or on edge',
    'PHQ4_4': 'Not being able to stop or control worrying',
}


BASE_LABELS = {
    'week_number': 'Fortnight index',
    'rolling_strength': '14-day mandate strength',
    'within_mandate_period': 'Mandate period indicator',
    'protective_behaviour_nomask_scale': 'Protective behaviour scale (excluding mask use)',
    'cantril_ladder': 'Life satisfaction ladder score',
    'age': 'Age',
    'i2_health': 'Number of close physical contacts',
    'r1_1': 'Perceived personal danger of COVID-19',
    'r1_2': 'Perceived likelihood of future COVID-19 infection',
}


PREFIX_LABELS = {
    'i9_health_': 'Would self-isolate if unwell',
    'i11_health_': 'Willingness to self-isolate for 7 days',
    'employment_status_': 'Employment status',
    'd1_comorbidities_': 'Comorbidities',
    'WCRex1_': 'View of government handling of COVID-19',
    'WCRex2_': 'Confidence in health system response',
}


VALUE_LABELS = {
    'N/A': 'No consent / missing by design',
    'Yes': 'Yes',
    'No': 'No',
    'Not sure': 'Not sure',
    'Very willing': 'Very willing',
    'Somewhat willing': 'Somewhat willing',
    'Neither willing nor unwilling': 'Neither willing nor unwilling',
    'Somewhat unwilling': 'Somewhat unwilling',
    'Very unwilling': 'Very unwilling',
    'A lot of confidence': 'A lot of confidence',
    'A fair amount of confidence': 'A fair amount of confidence',
    'Not very much confidence': 'Not very much confidence',
    'No confidence at all': 'No confidence at all',
    'Very well': 'Very well',
    'Somewhat well': 'Somewhat well',
    'Somewhat badly': 'Somewhat badly',
    'Very badly': 'Very badly',
    'Prefer_not_to_say': 'Prefer not to say',
    'Not working': 'Not working',
    'Retired': 'Retired',
    'Full time employment': 'Full-time employment',
    'Part time employment': 'Part-time employment',
    'Unemployed': 'Unemployed',
}


def prettify_value(raw_value):
    if raw_value in VALUE_LABELS:
        return VALUE_LABELS[raw_value]

    value = raw_value.replace('_', ' ')
    return value


def prettify_feature_name(feature_name):
    if feature_name in BASE_LABELS:
        return BASE_LABELS[feature_name]

    for phq_prefix, question_text in PHQ_LABELS.items():
        if feature_name.startswith(f'{phq_prefix}_'):
            response = feature_name.replace(f'{phq_prefix}_', '', 1)
            return f'{question_text}: {prettify_value(response)}'

    for prefix, label in PREFIX_LABELS.items():
        if feature_name.startswith(prefix):
            value = feature_name.replace(prefix, '', 1)
            return f'{label}: {prettify_value(value)}'

    return feature_name.replace('_', ' ')


def load_top_features_no_state(source_list, top_n=10):
    frames = []

    for source in source_list:
        if not os.path.exists(source['path']):
            print(f"Skipped missing file: {source['path']}")
            continue

        df = pd.read_csv(source['path'])
        df = df[~df['feature'].str.startswith('state_')].copy()
        df = df.sort_values('importance', ascending=False).head(top_n).copy()
        df['feature_label'] = df['feature'].apply(prettify_feature_name)
        df['period'] = source['label']
        frames.append(df[['feature', 'feature_label', 'importance', 'period']])

    if not frames:
        raise FileNotFoundError('No feature importance files were found.')

    return frames


def plot_feature_importance_panels_no_state(source_key, output_path, title_text):
    frames = load_top_features_no_state(FEATURE_IMPORTANCE_SOURCES[source_key])
    panel_count = len(frames)
    colors = ['#4E79A7', '#F28E2B', '#59A14F']

    fig, axes = plt.subplots(1, panel_count, figsize=(14, 6), constrained_layout=True)

    if panel_count == 1:
        axes = [axes]

    for i, df in enumerate(frames):
        df = df.sort_values('importance', ascending=True)
        axes[i].barh(df['feature_label'], df['importance'], color=colors[i])
        axes[i].set_title(df['period'].iloc[0])
        axes[i].set_xlabel('Feature Importance')
        axes[i].set_ylabel('')

    fig.suptitle(title_text, fontsize=14, fontweight='bold')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    ensure_output_dir()

    plot_feature_importance_panels_no_state(
        source_key='face_mask',
        output_path=FACE_MASK_OUTPUT,
        title_text='Top XGBoost Predictors of Face Mask Behaviour (Excluding State Variables)',
    )

    plot_feature_importance_panels_no_state(
        source_key='protective',
        output_path=PROTECTIVE_OUTPUT,
        title_text='Top XGBoost Predictors of Protective Behaviour (Excluding State Variables)',
    )

    print('Saved figures:')
    print(FACE_MASK_OUTPUT)
    print(PROTECTIVE_OUTPUT)


if __name__ == '__main__':
    main()
