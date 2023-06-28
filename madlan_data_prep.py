import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import re
import string
from sklearn.preprocessing import OneHotEncoder


path = "C:\\Users\\97250\\Desktop\\תואר הנדסה\\שנה ג' סמסטר ב\\data mining\\final_project\\output_all_students_Train_v8.xlsx"
df = pd.read_excel(path)


def prepare_data(df):
    df = df.iloc[:, :-1]
    df.columns = df.columns.str.strip()

    def get_numeric_price(string):
        string = str(string)
        price = ""
        for char in string:
            if char.isnumeric():
                price += char
            else:
                break
        if len(price) == 0:
            return None
        else:
            return int(price.replace(",", ""))

    df['price'] = df.price.astype(str)
    df['price'] = df['price'].apply(get_numeric_price)
    df.dropna(subset=['price'], inplace=True)
    df['price'] = df['price'].astype(int)
    df['Area'] = df['Area'].apply(lambda x: re.sub(r'[^0-9]', '', str(x)))
    df = df[df['Area'] != '']
    df['Area'] = df['Area'].astype(int)
    for column in df.columns:
        if column not in ['Area', 'price', 'room_number']:
            df[column] = df[column].astype(str).str.replace(r'[{}]'.format(re.escape(string.punctuation)), '',
                                                            regex=True)
    df['floor'] = df['floor_out_of'].str.extract(r'קומה (\d+)', expand=False)
    df['floor'] = df['floor'].fillna(0)
    df['floor'] = df['floor'].astype(int)
    df['total_floors'] = df['floor_out_of'].str.extract(r'מתוך (\d+)', expand=False)
    df['total_floors'] = df['total_floors'].fillna(0)
    df['total_floors'] = df['total_floors'].astype(int)

    def cat_entrance_date(data):
        if data == 'גמיש':
            return 'flexible'
        elif data == 'מיידי':
            return 'immediate'
        elif data == 'לא צויין':
            return 'not_defined'
        elif data.endswith('000000'):
            return 'above_year'
        elif pd.to_datetime(data, errors='coerce') >= pd.to_datetime('today') - pd.DateOffset(months=12):
            return 'months_6_12'
        else:
            return 'less_than_6 months'

    # Apply the cat_entrance_date function to each row of the 'entranceDate' column
    df['entranceDate'] = df['entranceDate'].apply(cat_entrance_date)
    df['entranceDate'] = df['entranceDate'].astype('category')

    def to_bool(string):
        string = str(string)
        if string.isnumeric():
            return string
        if 'יש' in string:
            return True
        elif 'כן' in string:
            return True
        elif 'yes' in string.lower():
            return True
        elif 'לא נגיש לנכים' in string:
            return False
        elif 'נגיש' in string:
            return True
        elif 'true' in string.lower():
            return True
        else:
            return False

    columns = ['hasAirCondition', 'hasBalcony', 'hasMamad', 'handicapFriendly']
    for column in columns:
        df[column] = df[column].apply(to_bool)
        df[column] = df[column].astype(bool)
        df[column] = df[column].astype(int)

    def extract_room_number(string):
        string = str(string)
        room_num = ""
        for char in string:
            if char.isnumeric() or char == '.':
                room_num += char
        return room_num

    df['room_number'] = df['room_number'].apply(extract_room_number)
    df['room_number'] = pd.to_numeric(df['room_number'], errors='coerce')
    df['room_number'] = df['room_number'].fillna(df['room_number'].median())
    df.iloc[53, 2] = 3.5  # instead of 35
    return df





df = prepare_data(df)







                    