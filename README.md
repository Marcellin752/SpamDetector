# SPAM DETECTOR — Pipeline Machine Learning *From Scratch*

Ce projet implémente un système complet de détection et de classification de spams SMS. Développé dans le cadre de l'association **Epi'AI**, l'objectif principal est de s'affranchir des bibliothèques de haut niveau (comme Scikit-Learn) pour l'ensemble du prétraitement textuel et de la modélisation mathématique, afin de comprendre, concevoir et d'entraîner des algorithmes de Machine Learning **entièrement *from scratch* avec NumPy**.

---

## Fonctionnalités
* **Pipeline de Prétraitement Manuel** : Nettoyage textuel, tokenisation personnalisée et calcul des scores **TF-IDF** codés sans boîte noire.
* **Train/Test Split Personnalisé** : Séparation aléatoire et étanche des données d'entraînement et de test écrite en NumPy.
* **Double Architecture *From Scratch*** :
  * **Régression Logistique** : Optimisation par descente de gradient binaire et calcul de l'Entropie Croisée (*Log Loss*).
  * **Multinomial Naïve Bayes** : Modélisation probabiliste basée sur le théorème de Bayes avec lissage de Laplace et calculs dans le domaine logarithmique.
* **Interfaces Web Streamlit** : Deux applications interactives distinctes permettant de tester les deux modèles en temps réel sur des messages inédits.

---

## Détails des Modèles Implémentés

### 1. Régression Logistique (`app.py`)

Ce modèle calcule une combinaison linéaire des scores TF-IDF du message, puis projette le résultat à travers la fonction d'activation **Sigmoïde** pour obtenir une probabilité entre 0 et 1 :
$$z = X \cdot W + b \quad \longrightarrow \quad \hat{y} = \frac{1}{1 + e^{-z}}$$
L'entraînement effectue une mise à jour manuelle des poids via le calcul des gradients de la fonction de coût *Log Loss*.

### 2. Multinomial Naïve Bayes (`app2.py`)
Ce modèle repose sur les probabilités conditionnelles et le théorème de Bayes. Pour éviter le sous-dépassement numérique (*underflow*), le calcul s'effectue dans le domaine logarithmique avec l'application systématique du lissage de Laplace ($\alpha = 1$) :
$$\log P(c \mid D) \propto \log P(c) + \sum_{i=1}^{n} X_i \cdot \log P(w_i \mid c)$$
Les scores logarithmiques bruts sont normalisés via une fonction **Softmax** pour renvoyer un pourcentage de confiance clair sur l'interface.

---

## Installation et Utilisation

### 1. Cloner le projet et configurer l'environnement
```bash
# Clonage du dépôt
git clone git@github.com:EPI-AI/SpamDetector.git
cd SpamDetector

# Création et activation de l'environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installation des dépendances nécessaires
pip install -r requirements.txt
```

### 2. Exécuter les applications Streamlit

Pour lancer l'interface connectée au modèle de **Régression Logistique** :
```bash
streamlit run apps/app.py
```

Pour lancer l'interface connectée au modèle Naïve Bayes :

```bash
streamlit run apps/app2.py
```

## Auteur

[Ipamma Marcellin SAMBIENI](https://github.com/Marcellin752) - Étudiant en Expertise Informatique à Epitech Bénin et membre de l'association Epi'AI

Peniel YAYI

Carlos SOSSOU