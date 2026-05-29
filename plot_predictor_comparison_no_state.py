import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_PATH = './data/new_data'
FIGURE_DIR = f'{BASE_PATH}/figures'
OUTPUT_PATH = f'{FIGURE_DIR}/predictor_comparison_no_state.png'

FEATURE_IMPORTANCE_SOURCES = {
    'Face mask behaviour': [
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
    'Protective behaviour': [
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

PERIOD_ORDER = ['All-time', 'Non-mandate', 'Mandate']
PERIOD_COLORS = {
    'All-time': '#4E79A7',
    'Non-mandate': '#F28E2B',
    'Mandate': '#59A14F',
}


def ensure_output_dir():
    os.makedirs(FIGURE_DIR, exist_ok=True)


SHORT_LABELS = {
    'week_number': 'Fortnight index',
    'rolling_strength': 'Mandate strength',
    'within_mandate_period': 'Mandate period',
    'protective_behaviour_nomask_scale': 'Protective behaviour (no mask)',
    'cantril_ladder': 'Life satisfaction',
    'age': 'Age',
    'i2_health': 'Close physical contacts',
    'r1_1': 'Perceived COVID danger',
    'r1_2': 'Perceived future infection risk',
    'i9_health_Yes': 'Would self-isolate if unwell: Yes',
    'i9_health_Not sure': 'Would self-isolate if unwell: Not sure',
    'i11_health_Very willing': 'Very willing to self-isolate',
    'i11_health_Somewhat willing': 'Somewhat willing to self-isolate',
    'i11_health_Very unwilling': 'Very unwilling to self-isolate',
    'employment_status_Not working': 'Employment: Not working',
    'employment_status_Retired': 'Employment: Retired',
    'd1_comorbidities_Yes': 'Comorbidities: Yes',
    'd1_comorbidities_Prefer_not_to_say': 'Comorbidities: Prefer not to say',
    'WCRex2_A lot of confidence': 'Health system confidence: High',
    'WCRex2_No confidence at all': 'Health system confidence: None',
    'WCRex1_Very well': 'Government handling: Very well',
    'PHQ4_1_Nearly every day': 'PHQ1: Nearly every day',
    'PHQ4_2_N/A': 'PHQ2: No consent / missing',
    'PHQ4_3_N/A': 'PHQ3: No consent / missing',
    'PHQ4_4_Nearly every day': 'PHQ4: Nearly every day',
}


def shorten_feature_name(feature_name):
    if feature_name in SHORT_LABELS:
        return SHORT_LABELS[feature_name]
    return feature_name.replace('_', ' ')


def load_group_data(source_list, top_n_each=8):
    frames = []

    for source in source_list:
        path = source['path']
        if not os.path.exists(path):
            print(f"Skipped missing file: {path}")
            continue

        df = pd.read_csv(path)
        df = df[~df['feature'].str.startswith('state_')].copy()
        df = df.sort_values('importance', ascending=False).head(top_n_each).copy()
        df['period'] = source['label']
        frames.append(df[['feature', 'importance', 'period']])

    if not frames:
        raise FileNotFoundError('No feature importance files were found.')

    combined_df = pd.concat(frames, ignore_index=True)
    return combined_df


def build_comparison_table(source_list, top_n_total=8):
    combined_df = load_group_data(source_list)
    feature_order = (
        combined_df.groupby('feature')['importance']
        .max()
        .sort_values(ascending=False)
        .head(top_n_total)
        .index
        .tolist()
    )

    filtered_df = combined_df[combined_df['feature'].isin(feature_order)].copy()
    pivot_df = (
        filtered_df.pivot_table(
            index='feature',
            columns='period',
            values='importance',
            aggfunc='max',
            fill_value=0,
        )
        .reindex(index=feature_order)
        .reindex(columns=PERIOD_ORDER, fill_value=0)
    )

    pivot_df['label'] = [shorten_feature_name(feature) for feature in pivot_df.index]
    return pivot_df


def plot_grouped_horizontal_bars(ax, pivot_df, title):
    labels = pivot_df['label'].tolist()
    data = pivot_df[PERIOD_ORDER].to_numpy()
    n_features = len(labels)
    y = np.arange(n_features)
    bar_height = 0.22
    offsets = [-bar_height, 0, bar_height]

    for i, period in enumerate(PERIOD_ORDER):
        ax.barh(
            y + offsets[i],
            data[:, i],
            height=bar_height,
            color=PERIOD_COLORS[period],
            label=period,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Feature importance')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)


def main():
    ensure_output_dir()

    face_mask_df = build_comparison_table(FEATURE_IMPORTANCE_SOURCES['Face mask behaviour'])
    protective_df = build_comparison_table(FEATURE_IMPORTANCE_SOURCES['Protective behaviour'])

    fig, axes = plt.subplots(2, 1, figsize=(14, 13), constrained_layout=True)

    plot_grouped_horizontal_bars(
        axes[0],
        face_mask_df,
        'Face mask behaviour: top predictors excluding state variables',
    )
    plot_grouped_horizontal_bars(
        axes[1],
        protective_df,
        'Protective behaviour: top predictors excluding state variables',
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle('XGBoost predictor importance by policy period', fontsize=16, fontweight='bold')
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print('Saved figure:')
    print(OUTPUT_PATH)


if __name__ == '__main__':
    main()
