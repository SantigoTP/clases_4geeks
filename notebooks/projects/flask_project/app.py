from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Asegúrate de que el nombre del archivo pkl sea exactamente este
modelo = joblib.load('modelo_concreto.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # 1. Extraer datos usando .get() para evitar el error 400
            # Si el nombre en el HTML no coincide, aquí recibiremos un None
            datos_form = {
                'cement': request.form.get('cement'),
                'blast_furnace_salag': request.form.get('slag'), # Nombre corregido según notebook
                'fly_ash': request.form.get('fly_ash'),
                'water': request.form.get('water'),
                'superplasticizer': request.form.get('superplasticizer'),
                'coarse_aggregate': request.form.get('coarse_agg'),
                'fine_aggregate': request.form.get('fine_agg'),
                'age': request.form.get('age')
            }

            # 2. VALIDACIÓN: Verificar si algún campo llegó vacío (causa del Error 400)
            for clave, valor in datos_form.items():
                if valor is None or valor == '':
                    # Esto te dirá exactamente qué campo falta en el HTML
                    return render_template('index.html', 
                                         prediction_text=f'Error: El campo "{clave}" no se recibió. Revisa el "name" en tu HTML.')

            # 3. Convertir a flotantes para el cálculo
            cement = float(datos_form['cement'])
            slag = float(datos_form['blast_furnace_salag'])
            fly_ash = float(datos_form['fly_ash'])
            water = float(datos_form['water'])
            superplasticizer = float(datos_form['superplasticizer'])
            coarse_agg = float(datos_form['coarse_aggregate'])
            fine_agg = float(datos_form['fine_aggregate'])
            age = float(datos_form['age'])

            # 4. Transformación logarítmica (Procesamiento de tu notebook)
            log_age = np.log(age) if age > 0 else 0

            # 5. Crear DataFrame con nombres de columnas exactos del entrenamiento
            input_data = pd.DataFrame([[
                cement, slag, fly_ash, water, 
                superplasticizer, coarse_agg, fine_agg, log_age
            ]], columns=[
                'cement', 'blast_furnace_salag', 'fly_ash', 'water', 
                'superplasticizer', 'coarse_aggregate', 'fine_aggregate', 'log_age'
            ])

            # 6. Predicción
            prediccion = modelo.predict(input_data)[0]

            return render_template('index.html', 
                                   prediction_text=f'La resistencia estimada es: {prediccion:.2f} MPa',
                                   datos_previos=request.form)

        except Exception as e:
            # Captura errores de conversión (letras en vez de números) o fallos del modelo
            return render_template('index.html', 
                                   prediction_text=f'Error detallado: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)