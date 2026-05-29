import pandas as pd


OXCGRT_PATH = './data/OxCGRT_AUS_latest.csv'
OUTPUT_PATH = './data/new_data/strong_mad_daate.csv'
BLOCK_DAYS = 14
MANDATE_THRESHOLD = 3


def build_mandate_table():
    source_df = pd.read_csv(OXCGRT_PATH)

    mandate_df = pd.DataFrame({
        'Date': pd.to_datetime(source_df['Date'], format='%Y%m%d'),
        'state': source_df['RegionCode'].str.replace('AUS_', '', regex=False),
        'mask_score': source_df['H6M_Facial Coverings'],
    })

    # Keep the original row order so the final output can be restored exactly
    # after block-level calculations.
    mandate_df['row_order'] = range(len(mandate_df))

    state_frames = []

    for _, state_df in mandate_df.groupby('state', sort=False):
        # Sort the whole state-level dataframe by date before assigning each
        # observation to a 14-day block. This changes entire rows together,
        # not a single column in isolation.
        ordered_state_df = state_df.sort_values('Date').copy()
        ordered_state_df['day_number'] = range(len(ordered_state_df))
        ordered_state_df['block_id'] = ordered_state_df['day_number'] // BLOCK_DAYS

        block_summary = (
            ordered_state_df.groupby('block_id', as_index=False)
            .agg(
                block_start=('Date', 'min'),
                block_end=('Date', 'max'),
                rolling_strength=('mask_score', 'mean'),
            )
        )

        ordered_state_df = ordered_state_df.merge(
            block_summary,
            on='block_id',
            how='left',
        )

        state_frames.append(ordered_state_df)

    result_df = pd.concat(state_frames, ignore_index=True)
    result_df['within_mandate_period'] = (
        result_df['rolling_strength'] >= MANDATE_THRESHOLD
    ).astype(int)

    # Restore the original row order so dates still align naturally with any
    # downstream matching step.
    result_df = result_df.sort_values('row_order').drop(
        columns=['row_order', 'day_number']
    )

    return result_df


mandate_result_df = build_mandate_table()
mandate_result_df.to_csv(OUTPUT_PATH, index=False)
