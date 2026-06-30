import streamlit as st
import pickle
import string
import numpy as np
from collections import Counter

# Configuration graphique de la page
st.set_page_config(
    page_title="Spam Detector - Naïve Bayes",
    page_icon="📊",
    layout="centered"
)

# Chargement des artefacts sauvegardés
@st.cache_resource
def load_nb_artifacts():
    # Chargement des connaissances du modèle probabiliste (Priors et Log-Vraisemblances)
    with open('src/custom_naive_bayes_model.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    
    # Chargement du dictionnaire IDF de la Séance 2
    with open('src/idf_corpus.pkl', 'rb') as f:
        idf_corpus = pickle.load(f)
        
    return artifacts['class_log_prior'], artifacts['feature_log_prob'], artifacts['classes'], idf_corpus

try:
    class_log_prior, feature_log_prob, classes, idf_corpus = load_nb_artifacts()
except FileNotFoundError:
    st.error("Fichier 'custom_naive_bayes_model.pkl' ou 'idf_corpus.pkl' introuvable. Veillez à exécuter le Notebook d'entraînement au préalable.")
    st.stop()

# Fonction de nettoyage et de tokenisation (Standardisée sur le projet)
def clean_and_tokenize(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    return text.split()

# Pipeline de traitement et prédiction probabiliste
def predict_naive_bayes(text, class_log_prior, feature_log_prob, idf_dict):
    tokens = clean_and_tokenize(text)
    if not tokens:
        return None, None

    # Calcul des Fréquences de Termes (TF) du message saisi
    mots_counts = Counter(tokens)
    tf_message = {word: count / len(tokens) for word, count in mots_counts.items()}
    
    # Alignement strict avec la taille du vocabulaire (nombre de caractéristiques apprises)
    X_vector = np.zeros(feature_log_prob.shape[1])
    
    # Reconstruction de l'indexation du vocabulaire à partir du dictionnaire IDF
    global_vocabulary = list(idf_dict.keys())
    word_to_index = {word: idx for idx, word in enumerate(global_vocabulary)}
    
    for word, tf_score in tf_message.items():
        if word in word_to_index:
            idx = word_to_index[word]
            idf_score = idf_dict[word]
            X_vector[idx] = tf_score * idf_score

    # Évaluation du Théorème de Bayes dans le domaine Logarithmique
    # Formule : Score = log P(c) + somme( X_vector * log P(w_i | c) )
    score_ham = class_log_prior[0] + np.sum(X_vector * feature_log_prob[0])
    score_spam = class_log_prior[1] + np.sum(X_vector * feature_log_prob[1])

    # Conversion des scores Logarithmiques en probabilités réelles via Softmax
    # On soustrait le max pour éviter l'overflow numérique induit par np.exp
    scores = np.array([score_ham, score_spam])
    exp_scores = np.exp(scores - np.max(scores))
    probabilities = exp_scores / np.sum(exp_scores)
    
    proba_ham = probabilities[0]
    proba_spam = probabilities[1]
    
    # Détermination de la classe finale (0 = Ham, 1 = Spam)
    prediction_class = 1 if proba_spam > proba_ham else 0
    
    return prediction_class, proba_spam


# --- INTERFACE UTILISATEUR STREAMLIT ---
st.title("Détecteur de Spams - Naïve Bayes From Scratch")
st.write("Visualisez les prédictions basées sur le théorème des probabilités conditionnelles.")

# --- SECTION EXPLICATIVE ---
with st.expander(" Théorie sous-jacente : Le modèle Naïve Bayes", expanded=False):
    st.markdown("""
    ### Fonctionnement de l'inférence Probabiliste
    Ce modèle calcule la probabilité qu'un message appartienne à la classe **Spam** ou **Ham** en additionnant les fréquences d'apparitions de chaque mot apprises lors de l'entraînement.
    
    * **Calcul du score textuel** : 
    $$\\log P(c \\mid D) \\propto \\log P(c) + \\sum_{i=1}^{n} X_i \\cdot \\log P(w_i \\mid c)$$
    * **Lissage de Laplace** : Appliqué lors de la phase d'entraînement pour éviter qu'un mot inconnu n'annule la probabilité complète ou ne cause une erreur $\\log(0)$.
    * **Normalisation (Softmax)** : Les scores bruts du modèle étant négatifs et très grands, nous utilisons une fonction d'activation exponentielle normalisée pour afficher des probabilités claires entre $0\\%$ et $100\\%$.
    """)

# Zone de saisie utilisateur
user_message = st.text_area("Saisissez le texte à évaluer :", placeholder="Collez votre SMS / Email ici...", height=150)

if st.button("Lancer la classification", type="primary"):
    if not user_message.strip():
        st.warning("Veuillez entrer un message valide.")
    else:
        # Exécution du modèle probabiliste
        pred_class, proba_spam = predict_naive_bayes(user_message, class_log_prior, feature_log_prob, idf_corpus)
        
        if pred_class is None:
            st.warning("Le message analysé ne contient aucun mot indexé dans le vocabulaire global.")
        else:
            st.markdown("### Métriques de classification :")
            
            # Affichage graphique des distributions de probabilité
            col1, col2 = st.columns(2)
            col1.metric(label="Confiance de classification SPAM", value=f"{proba_spam * 100:.2f} %")
            col2.metric(label="Confiance de classification HAM", value=f"{(1 - proba_spam) * 100:.2f} %")
            
            st.progress(float(proba_spam))
            
            # Diagnostic final
            if pred_class == 1:
                st.error("**RÉSULTAT : SPAM**")
                st.info("Le produit des probabilités conditionnelles des mots de ce message converge vers une signature de Spam.")
            else:
                st.success("**RÉSULTAT : HAM (MESSAGE SÛR)**")
                st.info("La distribution des termes correspond à la structure probabiliste des messages légitimes.")

st.markdown("---")
st.caption("Application connectée aux matrices de Log-vraisemblances NumPy du modèle Naïve Bayes. Projet Épi'AI.")
