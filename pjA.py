import pandas as pd
au = pd.read_csv('data/australia.csv')
au.info()
au.head(20)
au_clean = au.drop(columns=['RecordNo', 'endtime'], errors='ignore')

# Forward filter out the columns that we might need to use
health_cols = [col for col in au_clean.columns if 'health' in col]
demo_cols = [col for col in au_clean.columns if any(x in col for x in ['age','gender','state', 'household_size', 'employment_status'])]
vac_cols = [col for col in au_clean.columns if 'vac' in col]

# combine
combine_cols = ['qweek'] + health_cols + demo_cols
combine_cols_cols = [col for col in combine_cols if col in au_clean.columns]

# new dataset
au_c = au_clean[combine_cols]

# check new dataset
au_c.shape
au_c.head()
au_c.isna().sum().sort_values(ascending=False).head(20)
#The results show that, except for the 'child' -related variables, there are almost no missing values for the other variables
# The 'child' -related variables actually indicate whether there are children of different age groups in the family. Later, they can be combined into one column

# i12_health_21-25 represent the mask wearing in different condition, these values could combine in 1 column, and use mediumn 
mask_com = ['i12_health_21','i12_health_22','i12_health_23','i12_health_24','i12_health_25']
# median method of mask score
# au_c['mask_score_median'] = au_c[mask_com].median(axis=1)
# these 5 columns, first time always show no 'na' values but I cannot take any op, this is because of the some values show space
# so i need to switch this empty values to na and clean again
au_c[mask_com].isna().sum()
au_c[mask_com] = au_c[mask_com].replace(r'^\s*$', pd.NA, regex=True)
au_c['i12_health_21'].value_counts(dropna=False)
# string to num to bin
au_c['i12_health_22'].value_counts(dropna=False)
au_c['i12_health_23'].value_counts(dropna=False)
au_c['i12_health_24'].value_counts(dropna=False)
au_c['i12_health_25'].value_counts(dropna=False)
mask_value_map = {'Always':1,
                  'Frequently':2,
                  'Sometimes':3,
                  'Rarely':4,
                  'Not at all':5}
for col in mask_com:
    au_c[col] = au_c[col].astype('string').str.strip()
    au_c[col] = au_c[col].replace(mask_value_map)
    au_c[col] = pd.to_numeric(au_c[col], errors='coerce')

# median method 
au_c['mask_score_median'] = au_c[mask_com].median(axis=1)
au_c['mask_score_median'].value_counts(dropna=False)
# there are so many na values if we delete them, the amount of samples will reduce, and sample will lead some bias, because it could say only analyze those who are willing to answer
# result show 
au_c.head()

behaviour_cols_n = [
    'i12_health_2','i12_health_3','i12_health_4','i12_health_5',
    'i12_health_6','i12_health_7','i12_health_8','i12_health_9',
    'i12_health_10','i12_health_11','i12_health_12','i12_health_13',
    'i12_health_14','i12_health_15','i12_health_16','i12_health_17',
    'i12_health_18','i12_health_19','i12_health_20',
    'i12_health_26','i12_health_27','i12_health_28','i12_health_29'
] # 23 values
list(au_c.columns)
# drop cloumns include 'd1_', because these columns not important for model
au_c = au_c.drop(columns=[col for col in au_c.columns if 'd1_' in col])
au_c.info()
wrw = au_c.isna().sum()
#child and relatively values is not important so I delete them all
au_c = au_c.drop(columns=[col for col in au_c.columns if 'child_' in col])

mask_cols_n = [
    'i12_health_1', 'i12_health_21', 'i12_health_22', 'i12_health_23', 'i12_health_24', 'i12_health_25'
] # 6 values 
demographics_cols_n = [
    'age', 'gender', 'state', 'household_size', 'employment_status'
] # 5 values

# now we have 67 columns involve time value invlude 'qweek', and mask values include'
#########################################
au = pd.read_csv('data/australia.csv',
                 na_values = [' ', '__NA__'], keep_default_na = True, low_memory = False)
total_rows = len(au)
miss_count = au.isna().sum()
miss_persent = miss_count/total_rows
missing_df = pd.DataFrame({
    'column':au.columns,
    'miss_count':miss_count.values,
    'miss_persent':miss_persent.values
})
missing_df = missing_df.sort_values(by='miss_persent', ascending=True)
missing_df.to_csv('./data/new_data/missing_summary.csv', index=False)
# missing_summary
col_to_drop = missing_df[missing_df['miss_count']>10781]['column'].tolist()
au_fliter = au.drop(columns=col_to_drop)
au_fliter.to_csv('./data/new_data/au_fliter.csv', index=False)

# switch date 
au_fliter.head
