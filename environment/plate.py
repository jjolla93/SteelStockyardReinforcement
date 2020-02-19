import pandas as pd
import numpy as np
import scipy.stats as stats
import random
from datetime import datetime

random.seed = 42

def import_plates_schedule(filepath):
    df_schedule = pd.read_csv(filepath, encoding='euc-kr')
    plates = []
    for i, row in df_schedule.iterrows():
        plate = Plate(row['plate_id'], row['inbound_date'], row['outbound_date'])
        plates.append(plate)
    return plates


def import_plates_schedule_rev(filepath, graph=False):
    df_schedule = pd.read_csv(filepath, encoding='euc-kr')
    df_schedule.dropna(subset=['자재번호', '최근입고일', '블록S/C일자'], inplace=True)
    df_schedule['최근입고일'] = pd.to_datetime(df_schedule['최근입고일'], format='%Y.%m.%d')
    df_schedule['블록S/C일자'] = pd.to_datetime(df_schedule['블록S/C일자'], format='%Y.%m.%d')
    df_schedule = df_schedule[df_schedule['최근입고일'] >= datetime(2019, 1, 1)]
    df_schedule = df_schedule[df_schedule['최근입고일'] <= df_schedule['블록S/C일자']]
    initial_date = df_schedule['최근입고일'].min()
    df_schedule['최근입고일'] = (df_schedule['최근입고일'] - initial_date).dt.days
    df_schedule['블록S/C일자'] = (df_schedule['블록S/C일자'] - initial_date).dt.days
    df_schedule.sort_values(by=['최근입고일'], inplace=True)
    df_schedule.reset_index(drop=True, inplace=True)

    if graph:
        inter_arrival_time = (df_schedule['최근입고일'].diff()).dropna()
        stock_time = (df_schedule['블록S/C일자'] - df_schedule['최근입고일'])[df_schedule['블록S/C일자'] >= df_schedule['최근입고일']]
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        ax1.set_title('Inter Arrival Time'); ax1.set_xlabel('time'); ax1.set_ylabel('normalized frequency of occurrence')
        ax2.set_title('Stock Time'); ax2.set_xlabel('time'); ax2.set_ylabel('normalized frequency of occurrence')
        ax1.hist(list(inter_arrival_time), bins=100, density=True)
        ax2.hist(list(stock_time), bins=100, density=True)
        plt.show()

    plates = [[]]
    for i, row in df_schedule.iterrows():
        plate = Plate(row['자재번호'], row['최근입고일'], row['블록S/C일자'])
        plates[0].append(plate)
    return plates


def generate_schedule(arrival_scale=1/0.27, stock_mean=44, stock_std=32.5, num_plate=250):
    df_schedule = pd.DataFrame(columns=['자재번호', '최근입고일', '블록S/C일자'])
    arrivals = stats.expon.rvs(scale=arrival_scale, size=num_plate)
    arrivals[0] = 0
    #stocks = stats.expon.rvs(scale=stock_scale, size=num_plate)
    stocks = stats.norm.rvs(loc=stock_mean, scale=stock_std, size=num_plate)
    current_date = 0
    plates = [[]]
    for i in range(num_plate):
        plate_id = 'plate' + str(i)
        inbound_date = current_date + arrivals[i]
        outbound_date = inbound_date + stocks[i]
        if inbound_date > outbound_date:
            outbound_date = inbound_date
        current_date = inbound_date
        plate = Plate(plate_id, inbound_date, outbound_date)
        plates[0].append(plate)
        #row = pd.Series([plate_id, inbound_date, outbound_date], index=['자재번호', '최근입고일', '블록S/C일자'])
        #df_schedule = df_schedule.append(row, ignore_index=True)
    return plates #df_schedule


def import_plates_schedule_by_week(filepath):
    df_schedule = pd.read_csv(filepath, encoding='euc-kr')
    df_schedule.dropna(subset=['자재번호', '최근입고일', '블록S/C일자'], inplace=True)
    df_schedule['최근입고일'] = pd.to_datetime(df_schedule['최근입고일'], format='%Y.%m.%d')
    df_schedule['블록S/C일자'] = pd.to_datetime(df_schedule['블록S/C일자'], format='%Y.%m.%d')
    df_schedule = df_schedule[df_schedule['최근입고일'] >= datetime(2019, 1, 1)]
    df_schedule = df_schedule[df_schedule['최근입고일'] <= df_schedule['블록S/C일자']]
    initial_date = df_schedule['최근입고일'].min()
    df_schedule['최근입고일'] = (df_schedule['최근입고일'] - initial_date).dt.days
    df_schedule['블록S/C일자'] = (df_schedule['블록S/C일자'] - initial_date).dt.days
    df_schedule.sort_values(by=['블록S/C일자'], inplace=True)
    df_schedule.reset_index(drop=True, inplace=True)

    plates = []
    day = df_schedule['블록S/C일자'].min()
    while len(df_schedule) != 0:
        plates_by_week = []
        temp = df_schedule[df_schedule['블록S/C일자'] <= day]
        temp.sort_values(by=['최근입고일'], inplace=True)
        temp.reset_index(drop=True, inplace=True)
        steel_num = len(temp)

        if steel_num > 0:
            for i, row in temp.iterrows():
                # plate = Plate(row['자재번호'], row['최근입고일'], row['블록S/C일자'])
                plate = Plate(row['자재번호'], day - 7, row['블록S/C일자'])  # 주간 물량의 입고 기준일
                plates_by_week.append(plate)
            plates.append(plates_by_week)
            df_schedule.drop([_ for _ in range(steel_num)], inplace=True)
            df_schedule.reset_index(drop=True, inplace=True)
        random.shuffle(plates_by_week)
        day += 7

    return plates


def import_plates_schedule_by_day(filepath):
    df_schedule = pd.read_excel(filepath, header=[0, 1], encoding='euc-kr')
    columns = map(lambda x:x[0].replace('\n','') if 'Unnamed' in x[1] else x[0]+'_'+x[1], df_schedule.columns)
    df_schedule.columns = columns
    df_schedule.dropna(subset=['자재번호'], inplace=True)
    df_schedule['불출요구일'] = pd.to_datetime(df_schedule['불출요구일'], format='%Y.%m.%d')
    initial_date = df_schedule['불출요구일'].min()
    df_schedule['불출요구일'] = (df_schedule['불출요구일'] - initial_date).dt.days
    df_schedule.reset_index(drop=True, inplace=True)

    plates = []
    for (date, yard), group in df_schedule.groupby(['불출요구일', '적치장']):
        group.reset_index(drop=True, inplace=True)
        plates_by_day = []

        priority = 1
        while len(group) != 0:
            temp = group[group['절단장비'] == group.iloc[0]['절단장비']]
            steel_num = len(temp)
            for i, row in temp.iterrows():
                plate = Plate(row['자재번호'], date, date + priority)
                plates_by_day.append(plate)
            group.drop([_ for _ in range(steel_num)], inplace=True)
            group.reset_index(drop=True, inplace=True)
            priority += 1

        plates.append(plates_by_day)

    return plates


# 강재 정보 클래스 id, 입출고일정 포함
class Plate(object):
    def __init__(self, plate_id=None, inbound=0, outbound=1):
        self.id = str(plate_id)
        self.inbound = inbound
        self.outbound = outbound
        if outbound == -1:  # 강재 데이터가 없으면 임의로 출고일 생성
            self.outbound = random.randint(1, 5)


if __name__ == "__main__":
    inbounds = import_plates_schedule_by_day('../environment/data/강재+불출지시서.xlsx')
    length = [len(_) for _ in inbounds]
    print(np.max(length), np.argmax(length))
    '''
    f = open("../environment/data/plate.txt", 'w')
    for i in range(len(inbounds)):
        f.write(("-" * 50) + "\n")
        f.write("총 {0}개\n".format(len(inbounds[i])))
        f.write("\n")
        for j in range(len(inbounds[i])):
            f.write("자재번호: {0}, 최근입고일: {1}, 블록S/C일자: {2}\n".
                    format(inbounds[i][j].id, inbounds[i][j].inbound, inbounds[i][j].outbound))
    f.close()
    '''
