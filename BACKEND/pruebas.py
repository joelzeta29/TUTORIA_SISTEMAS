import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator

# Verificar el modelo
try:
    model = joblib.load('backend/models/emotion_model.pkl')
    if isinstance(model, BaseEstimator):
        print("✅ El modelo se cargó correctamente.")
    else:
        print("⚠️ El archivo 'emotion_model.pkl' no parece ser un modelo de scikit-learn.")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")

# Verificar el vectorizador
try:
    vectorizer = joblib.load('backend/models/vectorizer.pkl')
    if isinstance(vectorizer, TfidfVectorizer):
        print("✅ El vectorizador se cargó correctamente.")
    else:
        print("⚠️ El archivo 'vectorizer.pkl' no parece ser un TfidfVectorizer.")
except Exception as e:
    print(f"❌ Error al cargar el vectorizador: {e}")
