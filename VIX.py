#%% 讀取需要的模組
import pandas as pd
from scipy.interpolate import CubicSpline
from math import exp, sqrt, log
from tqdm.notebook import tqdm
## 使 tqdm 支持 Pandas
tqdm.pandas()
import warnings
warnings.filterwarnings('ignore', message="A value is trying to be set on a copy of a slice from a DataFrame")
# from IPython.display import Markdown

#%% 讀取資料
## 讀進來的資料是預先整理過到期日不大於 60 天的選擇權們
raw_data = pd.read_csv('./option_price.csv')
columns = raw_data.columns
## 對資料欄位做相應處理 (指定欄位型別可節省記憶體空間)
raw_data['date'] = pd.to_datetime(raw_data['date'])
raw_data['exdate'] = pd.to_datetime(raw_data['exdate'])
raw_data['cp_flag'] = raw_data['cp_flag'].map({'C': 1, 'P': 0}).astype(bool)
raw_data['am_settlement'] = raw_data['am_settlement'].astype(bool)
raw_data['week_end_flag'] = raw_data['week_end_flag'].astype(bool)
## 檢視資料
print(raw_data.info())
## date 為當天日期 (obj > datetime)
## strike 為履約價 (int)
## exdate 為到期日 (obj > datetime)
## cp_flag 為買、賣權標籤 C/P (obj > bool) 
## best_bid 為買方最高出價 (float)
## best_offer 為賣方最低報價 (float)
## am_settlement 為 SPX/SPXW 標籤 1/0 (int > bool)
## DTE 為距到期天數 (int)
## week_end_flag 為該選擇權 week-end 到期的標籤 1/0 (int > bool) (本欄是自行處理的資料，利用週五及期間唯一的週五假期 good friday 得出)
print(raw_data.head())

#%% 資料篩選
## 利用 week_end_flag 從整組資料中篩選週尾到期的選擇權
raw_data = raw_data[raw_data['week_end_flag'] == 1]
## 定義到期日篩選函數得到 near-term, next-term 資料集
def filter_dte_near(group):
    closest_below_30 = group[group['DTE'] <= 30]['DTE'].max()
    return group[group['DTE'] == closest_below_30].reset_index(drop=True)

def filter_dte_next(group):
    closest_above_30 = group[group['DTE'] >= 30]['DTE'].min()
    return group[group['DTE'] == closest_above_30].reset_index(drop=True)

near_data = raw_data.groupby('date').progress_apply(filter_dte_near).reset_index(drop=True)
next_data = raw_data.groupby('date').progress_apply(filter_dte_next).reset_index(drop=True)

print(near_data[['date', 'DTE']])
print(next_data[['date', 'DTE']])

#%% 計算內外插後的殖利率
## 讀進來的資料是原始 UST yield curve 殖利率 (單位為 %)
USTrate_data = pd.read_csv('./2019UST_Yield.csv')
## 對資料欄位做相應處理 (轉成日期格式、殖利率保留 1M, 2M 的即可)
USTrate_data['Date'] = pd.to_datetime(USTrate_data['Date'])
USTrate_data = USTrate_data.iloc[:, :3]
## 將兩筆資料 map 起來
df_near = near_data.merge(USTrate_data[['Date', '1 Mo', '2 Mo']], left_on='date', right_on='Date', how='left')
df_next = next_data.merge(USTrate_data[['Date', '1 Mo', '2 Mo']], left_on='date', right_on='Date', how='left')
## 創建 Natural Cubic Spline 計算函數
def CS_rate(row):
    r1, r2 = row['1 Mo'], row['2 Mo']
    cs = CubicSpline([30, 60], [r1, r2])
    r = log((1 + cs(row['DTE']) * 0.5) ** 2)
    return r
## 套用函數
df_near['CS_rate'] = df_near.progress_apply(CS_rate, axis=1)
df_next['CS_rate'] = df_next.progress_apply(CS_rate, axis=1)
## 刪除殖利率資料
df_near = df_near.drop(columns=['Date', '1 Mo', '2 Mo'])
df_next = df_next.drop(columns=['Date', '1 Mo', '2 Mo'])

print(df_near[['date', 'CS_rate']])
print(df_next[['date', 'CS_rate']])

#%% 計算 T (Time to Expiration)
## 創建TE計算函數
def TE(row):
    time_to_exp = row['DTE'] * 1440 if row['am_settlement'] else row['DTE'] * 1440 + 390
    return time_to_exp/525600
## 套用函數
df_near['T'] = df_near.progress_apply(TE, axis=1)
df_next['T'] = df_next.progress_apply(TE, axis=1)

print(df_near[['date', 'DTE', 'T']])
print(df_next[['date', 'DTE', 'T']])

#%% 計算 ATM Strike
## 首先計算 Mid Price
df_near['mid_price'] = (df_near['best_bid'] + df_near['best_offer']) * 0.5
df_next['mid_price'] = (df_next['best_bid'] + df_next['best_offer']) * 0.5
new_index = ['date', 'strike', 'exdate', 'cp_flag', 'best_bid', 'best_offer', 'mid_price', 'am_settlement', 'DTE', 'week_end_flag', 'CS_rate', 'T']
df_near = df_near.reindex(columns=new_index)
df_next = df_next.reindex(columns=new_index)
## 方法論有提到，成為 ATM candidate 的條件必須是 bid <= ask，或是 bid != 0
## 創建ATM_candidate_flag計算函數
def ATM_candidate_flag(row):
    ATM_candidate_flag = (row['best_bid'] <= row['best_offer']) & (row['best_bid'] != 0)
    return ATM_candidate_flag
## 套用函數
df_near['ATM_candidate_flag'] = df_near.progress_apply(ATM_candidate_flag, axis=1)
df_next['ATM_candidate_flag'] = df_next.progress_apply(ATM_candidate_flag, axis=1)
## 在計算同一天內 Call/Put mid_price 的最小差值
### 製作一個函數，輸入值為 df_near or df_next，輸出為一個新增 ATM, min_diff 欄位的 DataFrame
def gen_mindiff(df: pd.DataFrame) -> pd.DataFrame:
    # 將資料依照日期分組
    groupbydate_df = df.groupby('date')
    # 初始化空清單存放每組處理過的資料
    results = []
    # 對每組日期計算差值
    for date, group_df in tqdm(groupbydate_df):
        # 利用到 cp_flag 辨別買賣權，以及 ATM_candidate_flag 來篩過濾無法成為候選 ATM 的履約價
        call_df = group_df[(group_df['cp_flag'] == True) & (group_df['ATM_candidate_flag'] == True)]
        put_df = group_df[(group_df['cp_flag'] == False) & (group_df['ATM_candidate_flag'] == True)]
        if not call_df.empty and not put_df.empty:
            # 先算不同履約價下的絕對差值
            result = abs(call_df.set_index('strike')['mid_price'] - put_df.set_index('strike')['mid_price']).reset_index()
            # 另外算不同履約價下的買賣價差 (Call - Put)，並合併兩者
            mindiff_result = (call_df.set_index('strike')['mid_price'] - put_df.set_index('strike')['mid_price']).reset_index()
            result['min_diff'] = mindiff_result['mid_price']
            # 依照履約價升冪排序，因為取最小值時若相同的話會取道履約價最小那個，符合方法規定
            result = result.sort_values(by='strike', ascending=True)
            # 得到最小 Abs(Call - Put) 的index，並得到 ATM, (Call - Put)
            min_index = result['mid_price'].idxmin()
            ATM = result.loc[min_index, 'strike']
            min_diff = result.loc[min_index, 'min_diff']
            ATM_df = pd.DataFrame({'date': [date], 'ATM': [ATM], 'min_diff': [min_diff]})
            results.append(ATM_df)
    # 將結果合併為一筆資料
    final_result = pd.concat(results)
    # 接回原資料，利用日期做索引
    out_df = pd.merge(df, final_result, on=['date'], how='left')
    return out_df
### 利用函數計算每天的 ATM
df_near = gen_mindiff(df_near)
df_next = gen_mindiff(df_next)

print(df_near[['date', 'strike', 'mid_price', 'ATM', 'min_diff']])
print(df_next[['date', 'strike', 'mid_price', 'ATM', 'min_diff']])

#%% 計算 F、K_0 (Forward Price、strike price equal to or immediately below F)
## 創建Fwrd計算函數
def Fwrd(row):
    F = round(row['ATM'] + exp(row['CS_rate'] * row['T']) * row['min_diff'], 4)
    return F
## 套用函數
df_near['F'] = df_near.progress_apply(Fwrd, axis=1)
df_next['F'] = df_next.progress_apply(Fwrd, axis=1)
## 製作函數計算 K_0
def gen_K0(df: pd.DataFrame) -> pd.DataFrame:
    # 依照日期分組
    grouped = df.groupby('date')
    # 初始化清單，裝每天的 K_0
    K0_list = []    
    # 迭代每個日期
    for date, group in tqdm(grouped):
        # 獲得該天的 F，因為一天只有一個值，取第一個即可
        F = group['F'].iloc[0]
        # 找到當天小於 F 最大履約價
        K_0 = max(group[group['strike'] <= F]['strike'])
        # 創建一個索引資料集
        K0_df = pd.DataFrame({'date': [date], 'K_0': [K_0]})
        K0_list.append(K0_df)
    # 將結果合併為一筆資料
    final_result = pd.concat(K0_list)
    # 接回原資料，利用日期做索引
    out_df = pd.merge(df, final_result, on=['date'], how='left')
    return out_df
### 利用函數計算每天的 K_0
df_near = gen_K0(df_near)
df_next = gen_K0(df_next)

print(df_near[['date', 'strike', 'min_diff', 'F', 'K_0']])
print(df_next[['date', 'strike', 'min_diff', 'F', 'K_0']])

#%% Strike Selection & delta_K
## 製作函數計算 SS_flag & delta_K
def gen_SS_flag(df: pd.DataFrame) -> pd.DataFrame:
    # 依照日期分組
    grouped = df.groupby('date')
    # 初始化清單，裝每天的 K_0
    K0_list = []    
    # 迭代每個日期
    for date, group_df in tqdm(grouped):
        # 獲得該天的 K_0，因為一天只有一個值，取第一個即可
        K_0 = group_df['K_0'].iloc[0]
        # 利用到 cp_flag 辨別買賣權，以及 ATM_candidate_flag 來篩過濾無法成為候選 ATM 的履約價
        call_df = group_df[(group_df['cp_flag'] == True)].sort_values(by=['strike'], ascending=True).reset_index(drop=True)
        put_df = group_df[(group_df['cp_flag'] == False)].sort_values(by=['strike'], ascending=False).reset_index(drop=True)
        # 先遍歷 call_df: K 往上走
        SS_flag = []
        delta_K = []
        consec_0_count = 0
        for index, row in call_df.iterrows():
            # 當履約價大於等於 K_0，且未偵測到前面有連續兩次 0bid 者，才進入後續判定，否則貼標不採納
            if ((row['strike'] >= K_0) & (consec_0_count < 2)):
                # 當沒有出價時貼標不採納並記錄次數，用來偵測連續兩次 0bid
                if (row['best_bid'] == 0):
                    consec_0_count += 1
                    ans = False
                # 通過重重判定才貼採納標
                else:
                    ans = True
            else:
                ans = False
            SS_flag.append(ans)
            # 這個判定式目的在計算 delta_K，有 SS_flag == True 內的頭尾及中間兩種狀況
            if (ans):
                # 進入這個判定表示是 Selection 的下邊界 (頭)，delta_K = K_(i + 1) - K_i
                if (not SS_flag[index - 1]):
                    delta_K.append(call_df['strike'][index + 1] - call_df['strike'][index])
                # 進入這個判定表示是 Selection 的中間，delta_K = (K_(i + 1) - K_(i - 1)) / 2
                else:
                    delta_K.append((call_df['strike'][index + 1] - call_df['strike'][index - 1]) * 0.5)
            else:
                # 進入這個判定表示剛過 Selection 的上邊界 (尾)，修改上一個值為 delta_K = K_(i - 1) - K_(i - 2)，但本格為 0
                if (SS_flag[index - 1]):
                    delta_K[-1] = call_df['strike'][index - 1] - call_df['strike'][index - 2]
                # 進入這就是純不考慮的狀況
                delta_K.append(0)
        call_df['SS_flag'] = SS_flag
        call_df['delta_K'] = delta_K
        
        # 再遍歷 put_df: K 往下走
        SS_flag = []
        delta_K = []
        consec_0_count = 0
        for index, row in put_df.iterrows():
            # 當履約價小於等於 K_0，且未偵測到前面有連續兩次 0bid 者，才進入後續判定，否則一切為不採納
            if ((row['strike'] <= K_0) & (consec_0_count < 2)):
                if (row['best_bid'] == 0):
                    consec_0_count += 1
                    ans = False
                else:
                    ans = True
            else:
                ans = False
            SS_flag.append(ans)
            # 這個判定式目的在計算 delta_K，有 SS_flag == True 內的頭尾及中間兩種狀況
            if (ans):
                # 進入這個判定表示是 Selection 的下邊界 (頭)，delta_K = K_i - K_(i + 1)
                if (not SS_flag[index - 1]):
                    delta_K.append(put_df['strike'][index] - put_df['strike'][index + 1])
                # 進入這個判定表示是 Selection 的中間，delta_K = (K_(i - 1) - K_(i + 1)) / 2
                else:
                    # 2019/01/24 有遇到 Selection 邊界後剛好沒履約價的狀況，造成 KeyError。這邊採下邊界處理方式
                    try:
                        delta_K.append((put_df['strike'][index - 1] - put_df['strike'][index + 1]) * 0.5)
                    except KeyError:
                        delta_K.append(put_df['strike'][index - 1] - put_df['strike'][index])
            else:
                # 進入這個判定表示是 Selection 的上邊界 (尾)，delta_K = K_(i - 1) - K_i
                if (SS_flag[index - 1]):
                    delta_K[-1] = put_df['strike'][index - 2] - put_df['strike'][index - 1]
                # 進入這就是純不考慮的狀況
                delta_K.append(0)
        put_df['SS_flag'] = SS_flag
        put_df['delta_K'] = delta_K

        K0_list.append(pd.concat([call_df, put_df], ignore_index=True))

    # 將結果合併為一筆資料
    final_result = pd.concat(K0_list).sort_values(by=['date', 'cp_flag', 'strike'], ascending=[True, False, True])
    return final_result
### 利用函數產生每筆資料的 Strike_Seletion_flag
df_near = gen_SS_flag(df_near)
df_next = gen_SS_flag(df_next)

print(df_near[['date', 'SS_flag', 'delta_K']][(df_near['date'] == '2019-01-02') & (df_near['SS_flag'] == False)].head())
print(df_near[['date', 'SS_flag', 'delta_K']][(df_near['date'] == '2019-01-02') & (df_near['SS_flag'] == True)].head())

#%% 計算 Individual Contribution
## 創建Ind_Cntrb計算函數
def Ind_Cntrb(row):
    contribution = row['delta_K'] / (row['strike'] ** 2) * exp(row['CS_rate'] * row['T']) * row['mid_price']
    return contribution
## 套用函數
df_near['Contribution'] = df_near.progress_apply(Ind_Cntrb, axis=1)
df_next['Contribution'] = df_next.progress_apply(Ind_Cntrb, axis=1)

print(df_near[['date', 'strike', 'CS_rate', 'T', 'mid_price', 'delta_K', 'Contribution']][(df_near['SS_flag'] == True)])
print(df_next[['date', 'strike', 'CS_rate', 'T', 'mid_price', 'delta_K', 'Contribution']][(df_next['SS_flag'] == True)])

#%% 計算 Volatility
## 製作函數計算 Volatility
def Vol_Calc(df: pd.DataFrame) -> pd.DataFrame:
    # 依照日期分組
    grouped = df.groupby('date')
    # 初始化清單，裝每天的 Vol
    Vol_list = []
    # 迭代每個日期
    for date, group_df in tqdm(grouped):
        # 獲得該天的 T, F, K_0，因為一天只有一個值，取第一個即可
        T = group_df['T'].iloc[0]
        F = group_df['F'].iloc[0]
        K_0 = group_df['K_0'].iloc[0]
        # 將一天內的所有 Contribution 加總
        summation = sum(group_df['Contribution'])
        # 計算波動率
        volatility = 2 / T * summation - 1 / T * (F / K_0 - 1) ** 2
        # 創建一個索引資料集
        Vol_df = pd.DataFrame({'date': [date], 'Vol': [volatility]})
        Vol_list.append(Vol_df)
    # 將結果合併為一筆資料
    final_result = pd.concat(Vol_list).reset_index(drop=True)
    # 接回原資料，利用日期做索引
    out_df = pd.merge(df, final_result, on=['date'], how='left')
    return out_df
### 利用函數產生每筆資料的 Vol
df_near = Vol_Calc(df_near)
df_next = Vol_Calc(df_next)

print(df_near[['date', 'Vol']])
print(df_next[['date', 'Vol']])
# df_near.to_csv('./near_term.csv', index=False)
# df_next.to_csv('./next_term.csv', index=False)

#%% 合併資料計算 Interpolated Volatility
## 先貼 term 標籤，near/next = True/False
df_near['term_flag'] = True
df_next['term_flag'] = False
## 合併資料
df_all = pd.concat([df_near, df_next]).reset_index(drop=True)
# df_all.to_csv('./nn_term.csv', index=False)
# 依照日期分組
grouped = df_all.groupby('date')
# 初始化清單，裝每天的 VIX
VIX_list = []
# 迭代每個日期
for date, group_df in tqdm(grouped):
    # 利用到 cp_flag 辨別買賣權，以及 ATM_candidate_flag 來篩過濾無法成為候選 ATM 的履約價
    groupdf_near = group_df[(group_df['term_flag'] == True)].sort_values(by=['strike'], ascending=True).reset_index(drop=True)
    groupdf_next = group_df[(group_df['term_flag'] == False)].sort_values(by=['strike'], ascending=False).reset_index(drop=True)
    # 獲得該天的 T_1、T_2、Vol_1、Vol_2，因為一天只有一個值，取第一個即可
    T_1 = groupdf_near['T'].iloc[0]
    T_2 = groupdf_next['T'].iloc[0]
    Vol_1 = groupdf_near['Vol'].iloc[0]
    Vol_2 = groupdf_next['Vol'].iloc[0]
    M_cm = 60 * 24 * 30
    M_365 = 60 * 24 * 365
    M_t1 = T_1 * M_365
    M_t2 = T_2 * M_365
    # 計算 VIX
    if (M_t1 == M_t2):
        VIX = sqrt(Vol_1) * 100
    else:
        VIX = 100 * sqrt((T_1 * Vol_1 * ((M_t2 - M_cm) / (M_t2 - M_t1)) + T_2 * Vol_2 * ((M_cm - M_t1) / (M_t2 - M_t1))) * (M_365 / M_cm))
    # 創建一個索引資料集
    VIX_df = pd.DataFrame({'date': [date], 'VIX': [VIX]})
    VIX_list.append(VIX_df)
# 將結果合併為一筆資料
final_result = pd.concat(VIX_list).reset_index(drop=True)

print(final_result)
# final_result.to_csv('./VIX_result.csv', index=False)

#%% 驗證結果
## 針對 (Johnson, 2017) 建構結果進行比對
### 簡單資料處理，取 2019 前半年數據
JS_MKT_data = pd.read_csv('./vixts.csv').iloc[4:, :2]
JS_MKT_data.columns = ['date', 'Vol_JS']
JS_MKT_data['date'] = pd.to_datetime(JS_MKT_data['date'])
JS_MKT_data['Vol_JS'] = JS_MKT_data['Vol_JS'].astype(float)
JS_MKT_data = JS_MKT_data.loc[(JS_MKT_data['date'] >= '2019-01-02')].reset_index(drop=True)
### 計算 Vol -> VIX
def VIX(row):
    return (row['Vol_JS'] ** 0.5) * 100
JS_MKT_data['VIX_JS'] = JS_MKT_data.apply(VIX, axis=1)
# JS_MKT_data.info()
### 輸出相關性
correl = final_result['VIX'].corr(JS_MKT_data['VIX_JS'])
print(correl)
    
## 對歷史市場 VIX 資料進行相關性比對
### 簡單資料處理，取 2019 前半年數據
VIX_MKT_data = pd.read_csv('./^VIX.csv')[['Date', 'Adj Close']]
VIX_MKT_data.columns = ['date', 'VIX_MKT']
VIX_MKT_data['date'] = pd.to_datetime(VIX_MKT_data['date'])
VIX_MKT_data['VIX_MKT'] = VIX_MKT_data['VIX_MKT'].astype(str)
VIX_MKT_data = VIX_MKT_data[(VIX_MKT_data['VIX_MKT'] != 'nan')].reset_index(drop=True)
VIX_MKT_data['VIX_MKT'] = VIX_MKT_data['VIX_MKT'].astype(float)
# VIX_MKT_data.info()
### 輸出相關性
correl = final_result['VIX'].corr(VIX_MKT_data['VIX_MKT'])
print(correl)




































