import pandas as pd
from datetime import datetime


def convert_datetime(dt):
    date = dt.split()[0]
    return datetime.strptime(date, "%d/%m/%Y")


def household_convert(size_str):
    for i in range(1, 8):
        if size_str == str(i):
            return i
        elif size_str == "8 or more":
            return 8
        elif size_str == "Prefer not to say" or size_str == "Don't know":
            return None


au = pd.read_csv(
    "data/australia.csv",
    na_values=[" ", "__NA__"],
    keep_default_na=True,
)

total_rows = len(au)
miss_count = au.isna().sum()
miss_persent = miss_count / total_rows

missing_df = pd.DataFrame(
    {
        "column": au.columns,
        "miss_count": miss_count.values,
        "miss_persent": miss_persent.values,
    }
)
missing_df = missing_df.sort_values(by="miss_persent", ascending=True)
missing_df.to_csv("./data/new_data/missing_summary.csv", index=False)

thresh_value = 10781
col_to_drop = missing_df.loc[
    missing_df["miss_count"] > thresh_value, "column"
].tolist()

au_fliter = au.drop(columns=col_to_drop)
au_fliter.to_csv("./data/new_data/au_fliter.csv", index=False)

au_fliter["endtime"] = au_fliter["endtime"].apply(convert_datetime)

sdate = "2021-02-10"
edate = "2021-10-18"
window_time = (au_fliter["endtime"] <= edate) & (au_fliter["endtime"] >= sdate)

for i in range(1, 5):
    au_fliter.loc[window_time, f"PHQ4_{i}"] = au_fliter.loc[
        window_time, f"PHQ4_{i}"
    ].fillna("N/A")

for i in range(1, 14):
    au_fliter.loc[window_time, f"d1_health_{i}"] = au_fliter.loc[
        window_time, f"d1_health_{i}"
    ].fillna("N/A")

for i in range(98, 100):
    au_fliter.loc[window_time, f"d1_health_{i}"] = au_fliter.loc[
        window_time, f"d1_health_{i}"
    ].fillna("N/A")

au_fliter.dropna(inplace=True)

for i in range(1, 3):
    au_fliter[f"r1_{i}"] = au_fliter[f"r1_{i}"].replace(
        {
            "7 - Agree": 7,
            "6": 6,
            "5": 5,
            "4": 4,
            "3": 3,
            "2": 2,
            "1 – Disagree": 1,
            "1 - Disagree": 1,
        }
    )

frequency_dict = {
    "Always": 5,
    "Frequently": 4,
    "Sometimes": 3,
    "Rarely": 2,
    "Not at all": 1,
}

for column in au_fliter.columns:
    if column.startswith("i12_health_"):
        au_fliter[column] = au_fliter[column].map(frequency_dict)

face_mask_cols = [
    "i12_health_1",
    "i12_health_22",
    "i12_health_23",
    "i12_health_25",
]
au_fliter["face_mask_behaviour_scale"] = au_fliter[face_mask_cols].median(axis=1)
au_fliter["face_mask_behaviour_binary"] = au_fliter[
    "face_mask_behaviour_scale"
].apply(lambda x: "Yes" if x >= 4 else "No")

protective_behaviour_cols = [col for col in au_fliter if col.startswith("i12_")]
au_fliter["protective_behaviour_scale"] = au_fliter[
    protective_behaviour_cols
].median(axis=1)
au_fliter["protective_behaviour_binary"] = au_fliter[
    "protective_behaviour_scale"
].apply(lambda x: "Yes" if x >= 4 else "No")

protective_behaviour_nomask_cols = [
    col for col in protective_behaviour_cols if col not in face_mask_cols
]
au_fliter["protective_behaviour_nomask_scale"] = au_fliter[
    protective_behaviour_nomask_cols
].median(axis=1)

d1_cols = [col for col in au_fliter if col.startswith("d1_")]
au_fliter["d1_comorbidities"] = "Yes"
au_fliter.loc[au_fliter["d1_health_99"] == "Yes", "d1_comorbidities"] = "No"
au_fliter.loc[au_fliter["d1_health_99"] == "N/A", "d1_comorbidities"] = "NA"
au_fliter.loc[
    au_fliter["d1_health_98"] == "Yes", "d1_comorbidities"
] = "Prefer_not_to_say"

au_fliter = au_fliter.drop(d1_cols, axis=1)

start_date = au_fliter["endtime"].min()
au_fliter["week_number"] = ((au_fliter["endtime"] - start_date).dt.days // 14) + 1

au_fliter["household_size"] = au_fliter["household_size"].apply(household_convert)
au_fliter.dropna(inplace=True)

au_fliter = au_fliter.drop(
    ["qweek", "weight"] + protective_behaviour_cols,
    axis=1,
)

au_fliter.to_csv("./data/new_data/cleaned_au_fliter.csv", index=False)
au_fliter.info()
