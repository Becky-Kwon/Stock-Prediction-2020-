import pandas as pd
import numpy as np
from tuner import RFTuner
from sklearn.preprocessing import LabelEncoder


def fill_na(few_na_df):
    df = few_na_df.copy(deep=True)
    t = df.isna().sum() > 0
    t = t[t == True]
    na_columns = t.index.to_list()
    na_columns.append('업종코드')
    grouped_by_industry_code = df.groupby(['업종코드'], dropna=False).mean()

    for na_col in na_columns:
        # na가 아닌 항목들은 index_error를 일으켜서 해당 에러를 무시하면서 계속 돌도록
        try:
            na_mean_df = grouped_by_industry_code[na_col] # 업종코드와 na_col의 평균 값을 가지고 있는 DF
            comp_with_industry_codes = df[df[na_col].isna()]['업종코드']  # index: company_name, value: company_industry_code

            for company, code in zip(comp_with_industry_codes.index, comp_with_industry_codes.values):

                mean_value = na_mean_df.loc[code]

                # 평균 값이 nan이 아닐 때 산업부문 별 평균 값 넣고
                if not np.isnan(mean_value):
                    df.loc[company, na_col] = mean_value

                # 평균 값이 nan이면 해당 칼럼 중간값 넣기
                else:
                    df.loc[company, na_col] = df[na_col].median()
        except:
            pass
    return df


if __name__ == '__main__':

    pd.set_option('display.max_rows', 100)  # 최대 표시 줄 수 제한 해제
    pd.set_option('display.max_columns', None)  # 최대 표시 컬럼 수 제한 해제
    pd.set_option('display.max_colwidth', None)  # 컬럼내 데이터 표시 제한 해제

    '''t = pd.read_csv('./data/data_converted.csv', encoding='utf-8-sig')
    t = fill_na(t)
    t.to_csv('./data/data_filled.csv', encoding='utf-8-sig')'''


    df = pd.read_csv('./data/data_filled.csv', encoding='utf-8-sig')

    df = df.drop(['시가총액', '기업명'], axis=1)

    encoder = LabelEncoder()
    df['업종코드'] = encoder.fit_transform(df['업종코드'])
    df['업종코드'] = df['업종코드'].astype(np.object)

    pbounds = {
        'max_depth': (5, 30),
        'min_samples_split': (0.1, 0.5),
        'min_samples_leaf': (0.1, 0.5),
        'n_estimators': (2000, 2001)
    }

    tuner = RFTuner(df, 'target')
    tuner.train_model(pbounds, n_iter=50)