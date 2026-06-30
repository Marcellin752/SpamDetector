import streamlit as st
import pickle
import string
import numpy as np
from collections import Counter

# Configuration graphique de la page
st.set_page_config(
    page_title="Spam Detector App - From Scratch",
    page_icon="🛡️",
    layout="centered"
)

# Chargement des composants sauvegardés
@st.cache_resource
def load_artifacts():
    # Chargement des paramètres du modèle de Régression Logistique (poids et biais)
    with open('src/custom_logistic_model.pkl', 'rb') as f:
        model_artifacts = pickle.load(f)
    
    # Chargement du dictionnaire IDF
    with open('src/idf_corpus.pkl', 'rb') as f:
        idf_corpus = pickle.load(f)
        
    return model_artifacts['weights'], model_artifacts['bias'], idf_corpus

try:
    weights, bias, idf_corpus = load_artifacts()
except FileNotFoundError:
    st.error("Fichiers 'custom_logistic_model.pkl' ou 'idf_corpus.pkl' introuvables. Vérifiez votre dossier.")
    st.stop()

# Fonction de nettoyage et de tokenisation
def clean_and_tokenize(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    return text.split()

# Fonction de vectorisation et d'inférence mathématique From Scratch
def predict_message(text, weights, bias, idf_dict):
    tokens = clean_and_tokenize(text)
    
    if not tokens:
        return None

    # Calcul des Fréquences de Termes (TF) pour le message saisi
    mots_counts = Counter(tokens)
    tf_message = {word: count / len(tokens) for word, count in mots_counts.items()}
    
    # Calcul du score TF-IDF en alignement rigoureux avec l'ordre du vocabulaire d'entraînement
    # weights.shape[0] correspond au nombre exact de features (mots uniques du vocabulaire global)
    X_vector = np.zeros(weights.shape[0])
    
    # On reconstruit l'indexation du vocabulaire à partir des clés de idf_dict
    global_vocabulary = list(idf_dict.keys())
    word_to_index = {word: idx for idx, word in enumerate(global_vocabulary)}
    
    for word, tf_score in tf_message.items():
        if word in word_to_index:
            idx = word_to_index[word]
            idf_score = idf_dict[word]
            X_vector[idx] = tf_score * idf_score

    # --- Étape Inférence From Scratch (Mathématiques de la Régression Logistique) ---
    # Combinaison linéaire : z = X · W + b
    z = np.dot(X_vector, weights) + bias
    
    # Fonction d'activation Sigmoïde
    z = np.clip(z, -500, 500) # Évite les erreurs d'overflow géométriques
    proba_spam = 1 / (1 + np.exp(-z))
    
    return proba_spam

# --- Interface Utilisateur Streamlit ---
st.title("Détecteur de Spams - From Scratch")
st.write("Saisissez un message ci-dessous pour voir si c'est un message légitime (ham) ou non (spam).")

# Zone de saisie de texte
user_message = st.text_area("Message à analyser :", placeholder="Écrivez ou collez votre SMS/Email ici...", height=150)

if st.button("Analyser le message", type="primary"):
    if not user_message.strip():
        st.warning("Veuillez entrer un texte avant de lancer l'analyse.")
    else:
        # Lancement du pipeline complet
        proba = predict_message(user_message, weights, bias, idf_corpus)
        
        if proba is None:
            st.warning("Le message ne contient aucun mot valide après nettoyage.")
        else:
            # Seuil de décision classique à 0.5
            st.markdown("### Résultat de l'analyse :")
            
            # Affichage de jauges ou de métriques de confiance
            col1, col2 = st.columns(2)
            col1.metric(label="Probabilité que ce soit un SPAM", value=f"{proba * 100:.2f} %")
            col2.metric(label="Probabilité que ce soit un HAM", value=f"{(1 - proba) * 100:.2f} %")
            
            if proba >= 0.5:
                st.error("**SPAM DÉTECTÉ**")
                st.info("Ce message présente les caractéristiques typiques d'une tentative de phishing ou d'une publicité.")
            else:
                st.success("**MESSAGE LÉGITIME (HAM)**")
                st.info("Le message semble sûr et ne présente aucun comportement suspect.")
                
st.markdown("---")
st.caption("Application connectée aux paramètres NumPy (Poids & Biais) du modèle entraîné From Scratch.")
