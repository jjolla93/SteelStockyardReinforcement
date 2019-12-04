import pandas as pd
from random import randint


def import_plates_schedule(filepath):
    df_schedule = pd.read_csv(filepath, encoding='euc-kr')
    plates = []
    for i, row in df_schedule.iterrows():
        plate = Plate(row['plate_id'], row['inbound_date'], row['outbound_date'])
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
