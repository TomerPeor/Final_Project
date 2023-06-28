#Tomer Peor 315712422
#Noam Berns 203865779


import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import os
from sklearn.preprocessing import OneHotEncoder


columns = ['hasAirCondition', 'hasBalcony', 'hasMamad', 'handicapFriendly', 'Area', 'floor', 'room_number', 
           'total_floors', 'City_בית שאן', 'City_בת ים', 'City_גבעת שמואל', 'City_דימונה', 'City_הוד השרון', 'City_הרצליה', 
           'City_זכרון יעקב', 'City_חולון', 'City_חיפה', 'City_יהוד מונוסון', 'City_ירושלים', 'City_כפר סבא', 'City_מודיעין מכבים רעות', 
           'City_נהריה', 'City_נהרייה', 'City_נוף הגליל', 'City_נס ציונה', 'City_נתניה', 'City_פתח תקווה', 'City_צפת', 'City_קרית ביאליק', 
           'City_ראשון לציון', 'City_רחובות', 'City_רמת גן', 'City_רעננה', 'City_תל אביב', 'entranceDate_above_year', 
           'entranceDate_flexible', 'entranceDate_immediate', 'entranceDate_less_than_6 months', 'entranceDate_not_defined']
def cat_entrance_date(data):
    data = str(data)
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
app = Flask(__name__)
elastic_net = pickle.load(open('trained_model.pkl', 'rb'))

def to_model(final_features):
    new_df = pd.DataFrame(columns= columns, data=np.zeros(shape=(1,39)))
    col_city = f"City_{final_features[0]}"
    if col_city in new_df.columns:
        new_df.loc[0, col_city] = 1 
    entrence_date = cat_entrance_date(final_features[-1])
    col_entrence_date = f"entranceDate_{entrence_date}"
    if col_entrence_date in new_df.columns:
        new_df.loc[0, col_entrence_date] = 1 
    for i in range(0, 4):
        # new_df.iloc[0, i] = 1 if str(final_features[i+5]) == "TRUE" else 0
        new_df.iloc[0, i] = final_features[i+5]
    new_df.loc[0, "Area"] = int(final_features[2])
    new_df.loc[0, "room_number"] = int(final_features[1])
    new_df.loc[0, "floor"] = int(final_features[3])
    new_df.loc[0, "total_floors"] = int(final_features[4])
    return new_df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = request.form.getlist('feature')
    
    final_features = features

    df = to_model(final_features)

    price_pred = elastic_net.predict(df)[0]
    output_text = '{:,.2f}'.format(price_pred)
    output_text = output_text + " שח "

    return render_template('index.html', prediction_text='{}'.format(output_text))


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    
    app.run(host='0.0.0.0', port=port,debug=True)

