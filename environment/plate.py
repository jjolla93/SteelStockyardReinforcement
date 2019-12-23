import pandas as pd
from random import randint


def import_plates_schedule(filepath):
    df_schedule = pd.read_csv(filepath, encoding='euc-kr')
    plates = []
    for i, row in df_schedule.iterrows():
        plate = Plate(row['plate_id'], row['inbound_date'], row['outbound_date'])
        plates.append(plate)
    return plates

def import_plates_schedule_rev(filepath):
    df_schedule = pd.read_csv(filepath, encoding='euc-kr')
    plates = []
    num = df_schedule['자재번호']
    inbound_dates = pd.to_datetime(df_schedule['최근입고일'], format='%Y.%m.%d')
    outbound_dates = pd.to_datetime(df_schedule['블록S/C일자'], format='%Y.%m.%d')
    initial_date = inbound_dates.min()
    inbound_dates = inbound_dates - initial_date
    outbound_dates = outbound_dates - initial_date

    for i in range(len(num)):
        if inbound_dates[i] < outbound_dates[i]:
            plate = Plate(num[i], inbound_dates[i].days, outbound_dates[i].days)
            plates.append(plate)
    return plates


# 강재 정보 클래스 id, 입출고일정 포함
class Plate(object):
    def __init__(self, plate_id=None, inbound=0, outbound=1):
        self.id = str(plate_id)
        self.inbound = inbound
        self.outbound = outbound
        if outbound == -1:  # 강재 데이터가 없으면 임의로 출고일 생성
            self.outbound = randint(1, 5)
