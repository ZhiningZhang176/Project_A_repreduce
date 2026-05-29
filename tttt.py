import pandas as pd
from datetime import date

au = pd.read_csv(
    'data/australia.csv',
    na_values=[' ', '__NA__'],
    keep_default_na=True
)

# missing summary
total_rows = len(au)
miss_count = au.isna().sum()
miss_persent = miss_count / total_rows

missing_df = (
    pd.DataFrame({
        'column': au.columns,
        'miss_count': miss_count.values,
        'miss_persent': miss_persent.values
    })
    .sort_values(by='miss_persent', ascending=True)
)

missing_df.to_csv('./data/new_data/missing_summary.csv', index=False)

# drop columns with too many missing values
thresh_value = 10781
col_to_drop = missing_df.loc[
    missing_df['miss_count'] > thresh_value, 'column'
].tolist()

au_fliter = au.drop(columns=col_to_drop).copy()
au_fliter.to_csv('./data/new_data/au_fliter.csv', index=False)

# switch date
au_fliter['endtime'] = pd.to_datetime(
    au_fliter['endtime'],
    format='%d/%m/%Y %H:%M'
).dt.date

# special period
window_time = au_fliter['endtime'].between(date(2021, 2, 10), date(2021, 10, 18))

phq_cols = [f'PHQ4_{i}' for i in range(1, 5)]
chronic_cols = [f'd1_health_{i}' for i in range(1, 14)] + ['d1_health_98', 'd1_health_99']

au_fliter.loc[window_time, phq_cols] = au_fliter.loc[window_time, phq_cols].fillna('nn')
au_fliter.loc[window_time, chronic_cols] = au_fliter.loc[window_time, chronic_cols].fillna('nn')

# remove original missing values
au_fliter.dropna(inplace=True)

# agree scale
agree_dict = {
    '7 - Agree': 7,
    '6': 6,
    '5': 5,
    '4': 4,
    '3': 3,
    '2': 2,
    '1 – Disagree': 1,
    '1 - Disagree': 1
}

r_cols = [f'r1_{i}' for i in range(1, 3)]
for col in r_cols:
    au_fliter[col] = au_fliter[col].replace(agree_dict)

# frequency scale
frequency_dict = {
    'Always': 5,
    'Frequently': 4,
    'Sometimes': 3,
    'Rarely': 2,
    'Not at all': 1
}

i12_health_cols = [col for col in au_fliter.columns if col.startswith('i12_health_')]
for col in i12_health_cols:
    au_fliter[col] = au_fliter[col].map(frequency_dict)

# face mask behaviour
face_mask_cols = ['i12_health_1', 'i12_health_22', 'i12_health_23', 'i12_health_25']
au_fliter['face_mask_behaviour_scale'] = au_fliter[face_mask_cols].median(axis=1)
au_fliter['face_mask_behaviour_binary'] = au_fliter['face_mask_behaviour_scale'].ge(4).map({True: 'Yes', False: 'No'})

# protective behaviour
protective_behaviour_cols = [col for col in au_fliter.columns if col.startswith('i12_')]
au_fliter['protective_behaviour_scale'] = au_fliter[protective_behaviour_cols].median(axis=1)
au_fliter['protective_behaviour_binary'] = au_fliter['protective_behaviour_scale'].ge(4).map({True: 'Yes', False: 'No'})

protective_behaviour_nomask_cols = [
    col for col in protective_behaviour_cols if col not in face_mask_cols
]
au_fliter['protective_behaviour_nomask_scale'] = au_fliter[protective_behaviour_nomask_cols].median(axis=1)

# comorbidities
d1_cols = [col for col in au_fliter.columns if col.startswith('d1_')]
au_fliter['d1_comorbidities'] = 'Yes'
au_fliter.loc[au_fliter['d1_health_99'] == 'Yes', 'd1_comorbidities'] = 'No'
au_fliter.loc[au_fliter['d1_health_99'] == 'nn', 'd1_comorbidities'] = 'nn'
au_fliter.loc[au_fliter['d1_health_98'] == 'Yes', 'd1_comorbidities'] = 'Prefer_not_to_say'
au_fliter.drop(columns=d1_cols, inplace=True)

# week number
start_date = au_fliter['endtime'].min()
au_fliter['week_number'] = au_fliter['endtime'].apply(
    lambda x: ((x - start_date).days // 14) + 1
)

# household size
household_map = {str(i): i for i in range(1, 8)}
household_map['8 or more'] = 8
au_fliter['household_size'] = au_fliter['household_size'].map(household_map)

# remove induced missing values
au_fliter.dropna(inplace=True)

# drop columns not used in modelling
au_fliter.drop(columns=['qweek', 'weight'] + protective_behaviour_cols, inplace=True, errors='ignore')

# save cleaned data
au_fliter.to_csv('./data/new_data/cleaned_au_fliter.csv', index=False)
