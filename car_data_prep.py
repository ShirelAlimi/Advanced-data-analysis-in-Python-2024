import pandas as pd

def prepare_data(df):
    df = df[pd.to_datetime(df['Cre_date'], dayfirst=True, errors='coerce').notna()]
    df = df[pd.to_datetime(df['Repub_date'], dayfirst=True, errors='coerce').notna()]

    df['Cre_date'] = pd.to_datetime(df['Cre_date'], dayfirst=True)
    df['Repub_date'] = pd.to_datetime(df['Repub_date'], dayfirst=True)

    df['capacity_Engine'] = pd.to_numeric(df['capacity_Engine'], errors='coerce')
    df['Km'] = pd.to_numeric(df['Km'], errors='coerce')
    df['manufactor'] = df['manufactor'].replace('Lexsus', 'לקסוס')
    df['Gear'] = df['Gear'].replace('אוטומט', 'אוטומטית')
    df['Pic_num'] = df['Pic_num'].fillna(0).astype(int)

    categorical_columns = ['manufactor', 'model', 'Gear', 'Engine_type', 'Area', 'City']
    for col in categorical_columns:
        df[col] = df[col].astype('category')

    df['Current_Year'] = 2024
    df['age'] = df['Current_Year'] - df['Year']
    df['Km'] = df.groupby(['Year', 'Engine_type'])['Km'].transform(lambda x: x.fillna(x.mean()))
    df['capacity_Engine'] = df.groupby(['Engine_type', 'Year', 'model'])['capacity_Engine'].transform(lambda x: x.fillna(x.mean()))

    df.dropna(subset=['Km', 'Area', 'capacity_Engine', 'Gear'], inplace=True)
    columns_to_remove = ['Prev_ownership', 'Curr_ownership', 'Color', 'Supply_score', 'Test', 'Year', 'Current_Year']
    df.drop(columns=columns_to_remove, inplace=True)

    return df
