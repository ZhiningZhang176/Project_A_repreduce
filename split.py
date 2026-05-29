'''
Split data into training and testing sets for several modelling tasks.

This script reads the cleaned Australia dataset, merges a 14-day block-based
mask mandate strength indicator, creates dummy variables for categorical
columns, performs an 80/20 split, and saves model-specific X/y files.

Author:
    Zhang Zhining
'''

import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = './data/new_data/cleaned_au_fliter.csv'
MANDATE_PATH = './data/new_data/strong_mad_daate.csv'
OUTPUT_DIR = './data/new_data'
TEST_SIZE = 0.2
RANDOM_SEED = 20260417

STATE_NAME_MAP = {
    'Australian Capital Territory': 'ACT',
    'New South Wales': 'NSW',
    'Northern Territory': 'NT',
    'Queensland': 'QLD',
    'South Australia': 'SA',
    'Tasmania': 'TAS',
    'Victoria': 'VIC',
    'Western Australia': 'WA',
}

CATEGORICAL_COLUMNS = [
    'state',
    'gender',
    'i9_health',
    'employment_status',
    'i11_health',
    'WCRex1',
    'WCRex2',
    'PHQ4_1',
    'PHQ4_2',
    'PHQ4_3',
    'PHQ4_4',
    'd1_comorbidities',
]


def load_cleaned_data():
    df = pd.read_csv(DATA_PATH, keep_default_na=False)
    df['endtime'] = pd.to_datetime(df['endtime'], format='%Y-%m-%d')
    return df


def load_mandate_periods():
    mandate_df = pd.read_csv(MANDATE_PATH)
    mandate_df['Date'] = pd.to_datetime(mandate_df['Date'], format='%Y-%m-%d')
    return mandate_df


def add_mandate_period(df, mandate_df):
    merged_df = df.copy()
    merged_df['state_code'] = merged_df['state'].map(STATE_NAME_MAP)
    merged_df['survey_date'] = merged_df['endtime'].dt.normalize()

    period_df = mandate_df[['state', 'Date', 'rolling_strength', 'within_mandate_period']].copy()
    period_df = period_df.rename(
        columns={
            'state': 'state_code',
            'Date': 'survey_date',
        }
    )

    merged_df = merged_df.merge(
        period_df,
        on=['state_code', 'survey_date'],
        how='left',
    )

    merged_df['rolling_strength'] = merged_df['rolling_strength'].fillna(0.0)
    merged_df['within_mandate_period'] = merged_df['within_mandate_period'].fillna(0).astype(int)
    merged_df = merged_df.drop(columns=['state_code', 'survey_date'])

    return merged_df


def preprocess_data(df):
    processed_df = df.copy()
    dummy_df = pd.get_dummies(
        processed_df[CATEGORICAL_COLUMNS],
        prefix=CATEGORICAL_COLUMNS,
        drop_first=True,
        dtype=int,
    )
    processed_df = processed_df.drop(columns=CATEGORICAL_COLUMNS)
    processed_df = pd.concat([processed_df, dummy_df], axis=1)
    return processed_df


def save_split_files(train_df, test_df):
    train_df.to_csv(f'{OUTPUT_DIR}/df_train.csv', index=False)
    test_df.to_csv(f'{OUTPUT_DIR}/df_test.csv', index=False)


def encode_binary_target(series):
    return series.map({'No': 0, 'Yes': 1}).astype(int)


def build_xy_files(train_df, test_df, model_name, target_column, drop_columns, subset_mask=None):
    train_subset = train_df if subset_mask is None else train_df.loc[subset_mask(train_df)].copy()
    test_subset = test_df if subset_mask is None else test_df.loc[subset_mask(test_df)].copy()

    feature_columns = [col for col in train_df.columns if col not in drop_columns]

    x_train = train_subset.loc[:, feature_columns]
    x_test = test_subset.loc[:, feature_columns]
    y_train = encode_binary_target(train_subset[target_column])
    y_test = encode_binary_target(test_subset[target_column])

    x_train.to_csv(f'{OUTPUT_DIR}/X_train_{model_name}.csv', index=False)
    x_test.to_csv(f'{OUTPUT_DIR}/X_test_{model_name}.csv', index=False)
    pd.DataFrame({'y_train': y_train}).to_csv(
        f'{OUTPUT_DIR}/y_train_{model_name}.csv', index=False
    )
    pd.DataFrame({'y_test': y_test}).to_csv(
        f'{OUTPUT_DIR}/y_test_{model_name}.csv', index=False
    )


def non_mandate_mask(df):
    return df['within_mandate_period'] == 0


def mandate_mask(df):
    return df['within_mandate_period'] == 1


cleaned_df = load_cleaned_data()
mandate_periods = load_mandate_periods()
cleaned_df = add_mandate_period(cleaned_df, mandate_periods)
processed_df = preprocess_data(cleaned_df)
processed_df.to_csv(f'{OUTPUT_DIR}/cleaned_data_preprocessing.csv', index=False)

df_train, df_test = train_test_split(
    processed_df,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=processed_df['within_mandate_period'],
)

save_split_files(df_train, df_test)

common_drop_model_1 = [
    'RecordNo',
    'face_mask_behaviour_scale',
    'protective_behaviour_scale',
    'face_mask_behaviour_binary',
    'protective_behaviour_binary',
    'endtime',
]

common_drop_model_2 = [
    'RecordNo',
    'face_mask_behaviour_scale',
    'protective_behaviour_scale',
    'face_mask_behaviour_binary',
    'protective_behaviour_binary',
    'protective_behaviour_nomask_scale',
    'endtime',
]

build_xy_files(
    train_df=df_train,
    test_df=df_test,
    model_name='model_1',
    target_column='face_mask_behaviour_binary',
    drop_columns=common_drop_model_1,
)

build_xy_files(
    train_df=df_train,
    test_df=df_test,
    model_name='model_1a',
    target_column='face_mask_behaviour_binary',
    drop_columns=common_drop_model_1 + ['within_mandate_period'],
    subset_mask=non_mandate_mask,
)

build_xy_files(
    train_df=df_train,
    test_df=df_test,
    model_name='model_1b',
    target_column='face_mask_behaviour_binary',
    drop_columns=common_drop_model_1 + ['within_mandate_period'],
    subset_mask=mandate_mask,
)

build_xy_files(
    train_df=df_train,
    test_df=df_test,
    model_name='model_2',
    target_column='protective_behaviour_binary',
    drop_columns=common_drop_model_2,
)

build_xy_files(
    train_df=df_train,
    test_df=df_test,
    model_name='model_2a',
    target_column='protective_behaviour_binary',
    drop_columns=common_drop_model_2 + ['within_mandate_period'],
    subset_mask=non_mandate_mask,
)

build_xy_files(
    train_df=df_train,
    test_df=df_test,
    model_name='model_2b',
    target_column='protective_behaviour_binary',
    drop_columns=common_drop_model_2 + ['within_mandate_period'],
    subset_mask=mandate_mask,
)

print('preprocessed shape:', processed_df.shape)
print('train shape:', df_train.shape)
print('test shape:', df_test.shape)
