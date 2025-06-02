from flask import Flask, render_template, request
from ml_model import get_stock_data, prepare_data, train_model, predict_future_price
from datetime import datetime

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    stock = None
    date_str = None
    error = None
    if request.method == 'POST':
        stock = request.form['stock'].strip().upper()
        date_str = request.form['date']

        try:
            selected_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            today = datetime.today().date()
            days_ahead = (selected_date - today).days

            if days_ahead <= 0:
                error = "❌ Please choose a future date."
            elif days_ahead > 10:
                error = "⚠️ Prediction limited to 10 days ahead for this model."
            else:
                data = get_stock_data(stock)
                if data.empty or 'Close' not in data:
                    raise Exception("❌ Invalid or unsupported stock ticker.")
                data = prepare_data(data, days_ahead=days_ahead)
                model, _, _ = train_model(data)
                prediction = predict_future_price(model, data, days_ahead)

        except Exception as e:
            error = str(e)
    return render_template('index.html', prediction=prediction, error=error, stock=stock, date_str=date_str)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)
