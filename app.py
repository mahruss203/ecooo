from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Memuat model dan objek preprocessing
classifier = joblib.load('svm_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Mendapatkan data dari form
        mcg = float(request.form['mcg'])
        gvh = float(request.form['gvh'])
        lip = float(request.form['lip'])
        chg = float(request.form['chg'])
        aac = float(request.form['aac'])
        alm1 = float(request.form['alm1'])
        alm2 = float(request.form['alm2'])

        # Membuat array input
        x_baru = np.array([[mcg, gvh, lip, chg, aac, alm1, alm2]])

        # Preprocessing input
        x_baru = scaler.transform(x_baru)

        # Prediksi menggunakan model
        c2baru = classifier.predict(x_baru)
        prediction = label_encoder.inverse_transform(c2baru)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
