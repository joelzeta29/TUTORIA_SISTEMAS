import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Descargar recursos de NLTK si no están descargados
nltk.download('punkt')
nltk.download('stopwords')

# Función de preprocesamiento de texto
def preprocess_text(text):
    # Convertir a minúsculas
    text = text.lower()
    
    # Eliminar caracteres especiales y números
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenización
    tokens = word_tokenize(text)
    
    # Eliminar stopwords
    stop_words = set(stopwords.words('english'))  # Usa 'spanish' si los datos son en español
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# Cargar el dataset
data = pd.read_csv('e:/SEMESTRE 10 - 2/TRABAJO DE INVESTIGACION/TUTORIA_AVANZADA/BACKEND/data/emotion_data.csv', sep=';', encoding='ISO-8859-1')

# Preprocesar el texto
data['processed_text'] = data['text'].apply(preprocess_text)

# Dividir los datos en características (X) y etiquetas (y)
X = data['processed_text']
y = data['emotion']

# Definir las rutas correctas
vectorizer_path = 'e:/SEMESTRE 10 - 2/TRABAJO DE INVESTIGACION/TUTORIA_AVANZADA/BACKEND/models/vectorizer.pkl'
model_path = 'e:/SEMESTRE 10 - 2/TRABAJO DE INVESTIGACION/TUTORIA_AVANZADA/BACKEND/models/emotion_model.pkl'

# Crear un vectorizador TF-IDF y entrenarlo
vectorizer = TfidfVectorizer(max_features=5000)
X_vect = vectorizer.fit_transform(X)

# Guardar el vectorizador
with open(vectorizer_path, 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
print("Vectorizador guardado exitosamente.")

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Entrenar un modelo (por ejemplo, RandomForest)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Guardar el modelo entrenado
with open(model_path, 'wb') as model_file:
    pickle.dump(model, model_file)
print("Modelo entrenado y guardado exitosamente.")

# Realizar predicciones con el modelo entrenado
predictions = model.predict(X_test)

# Mostrar el informe de clasificación
print('Informe de clasificación:')
print(classification_report(y_test, predictions))

# Si deseas hacer una predicción para un nuevo texto:
new_text = ["me siento demasiado feliz, como para estar pensando en ti"] 
new_text_processed = [preprocess_text(text) for text in new_text]  # Preprocesar el nuevo texto
new_text_vect = vectorizer.transform(new_text_processed)
new_prediction = model.predict(new_text_vect)
print(f'Predicción para el nuevo texto: {new_prediction}')
