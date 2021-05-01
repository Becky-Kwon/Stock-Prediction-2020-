import pandas as pd
import os
import numpy as np


def transpose_df(company_info_df: pd.DataFrame, stock_name_with_csv: str):
    temp_dict = {}
    stock_name = stock_name_with_csv[:-4]

    try:
        standard_columns = ['2019.09', '2019.12', '2020.03', '2020.06', '2020.09']
        temp = company_info_df[standard_columns].copy()
        temp = temp.iloc[:-3]

        for col in standard_columns:
            total_columns = []
            transposed_col = temp[col].transpose()  # transpose
            s = pd.Series(transposed_col.index.values)  # make the index value to Series for applicating apply(lambda ) method
            transposed_col.index = s.apply(lambda x: f'{col}_{x}')
            total_columns = total_columns + [x for x in transposed_col.index]

            for refined_col in total_columns:
                temp_dict[refined_col] = [transposed_col[refined_col]]

        temp_df = pd.DataFrame.from_dict(temp_dict)
        temp_df.index = [stock_name]
        temp_df = convert_str_to_num(temp_df, stock_name)
        return temp_df

    except:
        return -1


def convert_str_to_num(_df, stock_name):
    df = _df.copy()

    for col in df.columns:
        if col != '시가총액':
            val = df.loc[stock_name, col]

            if isinstance(val, str):
                pos = val.find(',')

                while pos >= 0:
                    val = val[:pos] + val[pos + 1:]
                    pos = val.find(',')

                if val[0] == '-' and len(val) == 1:
                    val = 0

                df.loc[stock_name, col] = val
    return df


if __name__ == '__main__':
    pd.set_option('display.max_rows', 100)  # 최대 표시 줄 수 제한 해제
    pd.set_option('display.max_columns', None)  # 최대 표시 컬럼 수 제한 해제
    pd.set_option('display.max_colwidth', None)  # 컬럼내 데이터 표시 제한 해제

    df = pd.read_csv('./data/data.csv', encoding='utf-8-sig', index_col=0)
    object_cols = []

    for c in df.columns:
        if df[c].dtype == np.object:
            object_cols.append(c)

    for n in df.index:
        df = convert_str_to_num(df, n)

    for c in object_cols[1:]:
        df[c] = df[c].astype(np.float)

    df.to_csv('./data/data_converted.csv', encoding='utf-8-sig')














