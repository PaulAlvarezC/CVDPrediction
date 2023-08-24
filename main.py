from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

app = Flask(__name__)

DATA_FOLDER = 'data'
app.config['DATA_FOLDER'] = DATA_FOLDER
ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/getPredictionHeartDisease', methods=['POST'])
def getPredictionHeartDisease():
    if request.method == 'POST':
        if 'file' not in request.files:
            resp = jsonify({'message': 'No file part in the request', 'status': 400})
            resp.status_code = 400
            return resp
        
        files = request.files.getlist('file')
        errors = {}
        success = False

        for file in files:
            # Valido que el archivo sea correcto
            if file and allowed_file(file.filename):
                # Subo archivo en carpeta designada y coloco un nombre 
                # https://www.kaggle.com/datasets/adepvenugopal/heart-disease-data
                file.save(os.path.join(app.config['DATA_FOLDER'], 'heart_data.csv'))
                success = True
            else:
                errors['error'] = 'File type is not allowed'

        if success:
            # Cargando data en pandas data frame
            heart_data = pd.read_csv("data/heart_data.csv")
            print(heart_data.head(10))
            print(heart_data.shape)
            print(heart_data.describe())
            print(heart_data.info())
            # Verifico valores nulos
            print(heart_data.isnull().sum())
            # Verifico la distribución de la variable target
            print(heart_data['target'].value_counts())

            # Divido las demás caracteristicas y la variable target
            X = heart_data.drop(columns = 'target', axis = 1)
            print(X.head())
            # 'X' contiene los datos de la tabla sin incluir la columna TARGET, que se usara despues para el aprendizaje
            Y = heart_data['target']
            print(Y.head())
            # 'Y' contiene sola la columna TARGET para validar resultado antes de realizar el modelo

            # ******* Divido los datos para entrenamiento y pruebas
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.15, stratify = Y, random_state = 3 )
            # 1. stratify distribuira 0 y 1 de manera uniforme, la prediccion será imparcial
            # 2. test_split indica una proporción sobre el tamaño de los datos de prueba en el conjunto de datos, 
            # lo que significa que el 30 por ciento de los datos son datos de prueba
            # 3. random_state informa sobre la aleatoriedad de los datos, y el número informa sobre su grado de aleatoriedad

            # Verifico la forma de los datos divididos
            print(X.shape, X_train.shape, X_test.shape)

            # ******* Modelo de aprendizaje    Regresión logística
            model = LogisticRegression()
            model.fit(X_train.values, Y_train)
            LogisticRegression()

            # Modelo de Evaluación
            # Precisión de los datos de entrenamiento
            # La función de precisión mide la precisión entre dos valores o columnas
            X_train_prediction = model.predict(X_train.values)
            training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
            print("La precisión de los datos de entrenamiento : ", training_data_accuracy)

            # Precisión de los datos de prueba
            X_test_prediction = model.predict(X_test.values)
            test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
            print("La precisión de los datos de prueba : ", test_data_accuracy)


            # Sistema de predicción de edificios
            # Valores de características de entrada

            age = int(request.form['age'])
            gender = int(request.form['gender'])
            chestPain = int(request.form['chestPain'])
            trestbps = int(request.form['trestbps'])
            chol = int(request.form['chol'])
            fbs = int(request.form['fbs'])
            restecg = int(request.form['restecg'])
            thalach = int(request.form['thalach'])
            exang = int(request.form['exang'])
            oldpeak = int(request.form['oldpeak'])
            slope = int(request.form['slope'])
            ca = int(request.form['ca'])
            thal = int(request.form['thal'])


            # años, genero, cp, trestbps, colesterol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
            #  35     1      0     136       315       0     1        125     1       2        1     0    1

            input_data = (age, gender, chestPain, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
            print(input_data)
            # Cambio los datos de entrada en una matriz con numpy
            input_data_as_numpy_array = np.array(input_data)
            # Remodelo la matriz para predecir datos para una sola instancia
            reshaped_array = input_data_as_numpy_array.reshape(1,-1)

            # Predecir el resultado e imprimirlo
            prediction = model.predict(reshaped_array)
            print(prediction)
            
            # [0]: significa que el paciente tiene un corazón sano
            # [1]: significa que el paciente tiene un corazón enfermo

            if(prediction[0] == 0):
                result = "Paciente tiene un corazón saludable"
                print("Paciente tiene un corazón saludable 💛💛💛💛")
            else:
                result = "Paciente es propenso a tener problemas en el corazón"
                print("Paciente es propenso a tener problemas en el corazón 💔💔💔💔")

            #Genero la respuesta en JSON
            resp = jsonify({'message': 'CSV successfully upload', 'status': 200, 'result': result, 'input': input_data,})
            resp.status_code = 200
            return resp
        else:
            resp = jsonify({'message': errors, 'status': 500})
            resp.status_code = 500
            return resp

if __name__ == '__main__':
    app.run(debug=True, port=5000)