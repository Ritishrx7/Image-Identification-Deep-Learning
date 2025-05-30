# Image Identification Deep Learning

Projet de deep learning pour l'identification et la classification d’images de voitures Tesla à l’aide du transfer learning avec EfficientNetB3.

## Description

Le but de ce projet est d'utiliser EfficientNetB3 pour du transfer learning afin de prédire/classifier des images de Tesla.  
Ce projet démontre comment exploiter des réseaux de neurones profonds pré-entraînés pour améliorer les performances sur des jeux de données spécifiques à l’aide de PyTorch et Streamlit.

## Installation

Cloner le dépôt :
```bash
git clone https://github.com/Ritishrx7/Image-Identification-Deep-Learning.git
cd Image-Identification-Deep-Learning
```

Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

Ouvrez le notebook principal avec Jupyter :
```bash
jupyter notebook
```

Ou lancez l’interface Streamlit :
```bash
streamlit run DL_Streamlit_tesla.py
```

## Données

Les images utilisées pour l’entraînement et les tests se trouvent dans le dossier `assets`.

## Technologies utilisées

- Python, Jupyter Notebook
- PyTorch, torchvision
- EfficientNetB3 (transfer learning)
- scikit-learn, pandas, numpy, matplotlib, seaborn, Pillow
- Streamlit (pour l’interface utilisateur)

## Contribuer

Les contributions sont les bienvenues !  
N’hésitez pas à ouvrir une issue ou à soumettre une pull request.

## Auteur

- [Ritishrx7](https://github.com/Ritishrx7)

## Licence

Projet open source.
