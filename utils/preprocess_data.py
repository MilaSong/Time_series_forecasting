
import pandas as pd
import numba as nb
import numpy as np
from multiprocessing import Pool

grid = pd.read_csv('data_grid.csv', encoding='utf-8')
grid_arr = grid[['ID', 'x0', 'x2', 'y0', 'y2']].to_numpy()


def read_yearly_files(year_list):
    files = []
    for year in year_list:
        files.append(pd.read_csv(f'data/raw/FILE_{year}.csv', encoding='utf-8'))
    return files


def read_one_week(start='YYYY-mm-dd 00:00:00', end='YYYY-mm-dd 00:00:00'):
    ''' Used for generating test files '''
    dataframe = pd.read_csv('data/FILE_YYY.csv', encoding='utf-8')
    dataframe = dataframe[(dataframe['COL1'] >= start) & (dataframe['COL1'] < end)]
    return [dataframe]


# ======================================
# ====== Preparation stages, 10s =======
# ======================================

@nb.njit
def search_grids(X, Y):
    for tup in grid_arr:
        if tup[4] > X >= tup[3] and tup[2] > Y >= tup[1]:
            return tup[0]

@nb.njit
def data_preparation(koordsx, koordsy):
    return [search_grids(x,y) for x, y in zip(koordsx, koordsy)]


def prep_dataframe(df):
    type_list = ["CLASS1", "CLASS12", "CLASS3"]
    df = df[df.TYPE.isin(type_list)]
    df = df.dropna(subset=['X', 'Y'])
    
    koordsx = nb.typed.List(df.X)
    koordsy = nb.typed.List(df.Y)
    grids_ids = data_preparation(koordsx, koordsy)
    df['ID'] = grids_ids

    return df


def filter_categories():
    start_year = YYYY
    end_year = YYYY + x
    years_list = np.arange(start_year, end_year)
    files_list = read_yearly_files(years_list)

    with Pool(len(years_list)) as p:
        dfs = p.map(prep_dataframe, files_list)

    final_df = pd.concat(dfs)
    keys_list = ['CLASS', 'TYPE', 'X', 'Y', 'LNG', 'LAT', 'ID']
    final_df[keys_list].to_csv("data/preprocessed/events_data.csv", index=False)
