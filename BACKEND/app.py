from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

# Configuración de la app
app = Flask(__name__,
            template_folder=os.path.join(os.getcwd(), 'FRONTEND', 'templates'),  # Asegúrate de que Flask busque las plantillas aquí
            static_folder=os.path.join(os.getcwd(), 'FRONTEND', 'static'))  # Asegúrate de que Flask busque los archivos estáticos aquí
CORS(app)

# Intentar cargar el modelo y el vectorizador
try:
    model = joblib.load('backend/models/emotion_model.pkl')
    vectorizer = joblib.load('backend/models/vectorizer.pkl')
    print("✅ Modelo y vectorizador cargados correctamente.")
except Exception as e:
    print(f"❌ Error al cargar el modelo o el vectorizador: {e}")
    exit(1)

# Preguntas y recomendaciones
recommendations = {
    'feliz': "¡Qué bueno que te sientas feliz! Continúa haciendo lo que te hace sentir bien.",
    'ansioso': "Parece que estás un poco ansioso. Trata de relajarte con respiraciones profundas o una caminata.",
    'estresado': "El estrés es común. Asegúrate de tomar pausas y organizar tus tareas.",
    'triste': "Sentirse triste a veces es normal. Habla con alguien de confianza o busca actividades que te alegren.",
    'abrumado': "Estás abrumado. Organiza tus tareas y tómate un descanso si lo necesitas.",
    'enojo': "Parece que estás molesto. Intenta calmarte y tomar un momento para ti mismo.",
    'tranquilo': "Qué bien que te sientas tranquilo. Aprovecha esa calma para descansar y recargar energías."
}

# Función de preprocesamiento de texto
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Záéíóúü\s]', '', text)  # Permite caracteres en español
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('spanish'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Ruta principal
@app.route('/')
def home():
    return render_template('index.html')  # Flask buscará index.html en FRONTEND/templates

# Endpoint de predicción
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validar datos de entrada
        data = request.get_json()
        if 'input' not in data:
            return jsonify({'error': 'Falta el campo "input" en la solicitud'}), 400
        
        user_input = data['input']
        
        # Preprocesar la entrada
        processed_input = preprocess_text(user_input)
        
        # Vectorizar la entrada
        input_vector = vectorizer.transform([processed_input])
        
        # Realizar la predicción
        prediction = model.predict(input_vector)[0]
        
        # Obtener la recomendación correspondiente
        recommendation = recommendations.get(prediction, "No se pudo determinar un estado emocional.")
        
        return jsonify({'emotion': prediction, 'recommendation': recommendation})
    except Exception as e:
        return jsonify({"error": f"Error interno: {str(e)}"}), 500

# Endpoint para manejar interacciones con el chatbot
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if 'message' not in data:
            return jsonify({'error': 'Falta el campo "message" en la solicitud'}), 400

        user_message = data['message']
        
        # Preprocesar el mensaje
        processed_message = preprocess_text(user_message)
        
        # Vectorizar el mensaje
        message_vector = vectorizer.transform([processed_message])
        
        # Realizar la predicción
        prediction = model.predict(message_vector)[0]
        
        # Obtener la recomendación
        bot_response = recommendations.get(prediction, "No se pudo determinar un estado emocional.")
        
        return jsonify({'emotion': prediction, 'response': bot_response})
    except Exception as e:
        return jsonify({"error": f"Error interno: {str(e)}"}), 500

# Ejecutar la app
if __name__ == '__main__':
    app.run(debug=True, port=5000)