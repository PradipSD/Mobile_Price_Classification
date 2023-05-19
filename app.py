from flask import Flask, render_template, request
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
# Load the trained model
model = KNeighborsClassifier(n_neighbors=9)
train_data = pd.read_csv('updated_train_data.csv')


x_train = train_data.drop('price_range', axis=1)
y_train = train_data['price_range']
model.fit(x_train, y_train)

app = Flask(__name__)

# Load the trained model and feature names
model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train, y_train)

feature_names = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
                 'int_memory', 'mobile_wt', 'n_cores', 'pc', 'px_height',
                 'px_width', 'ram', 'sc_h', 'sc_w', 'three_g',
                 'touch_screen', 'wifi']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = []
    for feature in feature_names:
        value = request.form[feature]
        features.append(value)

    input_data = pd.DataFrame([features], columns=feature_names)
    predicted_price_range = model.predict(input_data)

    return render_template('index.html', prediction='Predicted price range: {}'.format(predicted_price_range[0]))


if __name__ == '__main__':
    app.run(debug=True,--host --port 10000)
