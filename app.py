import pandas as pd
from flask import Flask, render_template, request
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load and preprocess the dataset
file_path = "ecoli.csv"
df = pd.read_csv(file_path)

# Separate features and labels
feature_columns = ["mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2"]
X = df[feature_columns].values
y = df["class"].values  # Pastikan nama kolom kelas sesuai dengan dataset Anda

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM classifier
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input data from the form
        mcg = float(request.form['mcg'])
        gvh = float(request.form['gvh'])
        lip = float(request.form['lip'])
        chg = float(request.form['chg'])
        aac = float(request.form['aac'])
        alm1 = float(request.form['alm1'])
        alm2 = float(request.form['alm2'])

        # Make prediction for the input data
        x_new = [[mcg, gvh, lip, chg, aac, alm1, alm2]]
        x_new = scaler.transform(x_new)
        prediction = classifier.predict(x_new)
        predicted_class = label_encoder.inverse_transform(prediction)[0]

        train_prediction = classifier.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_prediction)

        test_prediction = classifier.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_prediction)

    return render_template('index.html', prediction=predicted_class, train_accuracy=train_accuracy, test_accuracy=test_accuracy)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
