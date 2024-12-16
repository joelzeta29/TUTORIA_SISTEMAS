import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# Descargar recursos de NLTK si aún no están descargados
nltk.download('punkt')
nltk.download('stopwords')

# Función de preprocesamiento de texto
def preprocess_text(text):
    # Convertir a minúsculas
    text = text.lower()
    
    # Eliminar caracteres especiales y números
    text = re.sub(r'[^a-zA-Záéíóúü\s]', '', text)  # Asegúrate de incluir caracteres en español
    
    # Tokenización
    tokens = word_tokenize(text)
    
    # Eliminar stopwords en español
    stop_words = set(stopwords.words('spanish'))  # Cargar stopwords en español
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# Cargar el dataset desde el archivo CSV con el delimitador correcto
dataset_path = r"E:\SEMESTRE 10 - 2\TRABAJO DE INVESTIGACION\TUTORIA_AVANZADA\BACKEND\data\emotion_data.csv"
try:
    # Aquí indicamos que el archivo usa punto y coma como delimitador
    data = pd.read_csv(dataset_path, encoding='latin1', delimiter=';')
except Exception as e:
    print(f"Error al cargar el archivo: {e}")
    exit()

# Mostrar las primeras filas del archivo para verificar que todo esté correcto
print(data.head())

# Asegurarse de que las columnas tengan los nombres correctos
data.columns = [col.strip().lower() for col in data.columns]  # Convertir a minúsculas y eliminar espacios adicionales

# Preprocesar los textos
data['text'] = data['text'].apply(preprocess_text)

# Vectorizador y modelo
vectorizer = TfidfVectorizer()
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Entrenar el modelo
X = data['text']
y = data['emotion']
model.fit(X, y)

# Guardar el modelo entrenado
joblib.dump(model, 'emotion_model.pkl')

# Recomendaciones para cada emoción
recommendations = {
    'feliz': "¡Qué bueno que te sientas feliz! Continúa haciendo lo que te hace sentir bien.",
    'ansioso': "Parece que estás un poco ansioso. Trata de relajarte con respiraciones profundas o una caminata.",
    'estresado': "El estrés es común. Asegúrate de tomar pausas y organizar tus tareas.",
    'triste': "Sentirse triste a veces es normal. Habla con alguien de confianza o busca actividades que te alegren.",
    'abrumado': "Estás abrumado. Organiza tus tareas y tómate un descanso si lo necesitas.",
    'enojo': "Parece que estás molesto. Intenta calmarte y tomar un momento para ti mismo.",
    'tranquilo': "Qué bien que te sientas tranquilo. Aprovecha esa calma para descansar y recargar energías."
}

# Función para predecir la emoción y proporcionar una recomendación
def predict_emotion(text):
    # Preprocesar el texto
    processed_text = preprocess_text(text)
    
    # Cargar el modelo
    model = joblib.load('emotion_model.pkl')
    
    # Predecir la emoción
    prediction = model.predict([processed_text])[0]
    print(f"Predicción: {prediction}")  # Muestra la predicción
    
    # Obtener recomendación
    recommendation = recommendations.get(prediction, "No hay recomendación disponible.")
    
    return {
        "emotion": prediction,
        "recommendation": recommendation
    }

# Ejemplo de uso
if __name__ == "__main__":
    text = input("Escribe tu estado emocional: ")
    result = predict_emotion(text)
    print(result)