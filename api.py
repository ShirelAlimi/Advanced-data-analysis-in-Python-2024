from flask import Flask, request, jsonify
import pandas as pd
import joblib
from car_data_prep import prepare_data

app = Flask(__name__)
model = joblib.load('trained_model.pkl')


@app.route('/estimate_price', methods=['POST'])
def estimate_price():
    # קבלת הנתונים מהטופס
    data = {
        'manufactor': request.form['manufactor'],
        'model': request.form['model'],
        'year': int(request.form['year']),
        'km': int(request.form['km']),
        'engine_capacity': float(request.form['engine_capacity']),
        'gear': request.form['gear'],
        'engine_type': request.form['engine_type'],
        'area': request.form['area'],
        'city': request.form['city']
    }

    # המרת הנתונים ל-DataFrame
    df = pd.DataFrame([data])

    # עיבוד הנתונים
    df = prepare_data(df)
    df = pd.get_dummies(df, drop_first=True)

    # הבטחת שהעמודות הדרושות קיימות
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0

    df = df[model.feature_names_in_]

    # חיזוי מחיר הרכב
    price = model.predict(df)

    return jsonify({'estimated_price': price[0]})


if __name__ == '__main__':
    app.run(debug=True)
