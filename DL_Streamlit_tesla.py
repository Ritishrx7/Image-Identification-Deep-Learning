import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import efficientnet_b3
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
import pickle

@st.cache_resource
def load_model():
    try:
        with open('model_config.pkl', 'rb') as f:
            config = pickle.load(f)
    except FileNotFoundError:
        st.warning("Model configuration file not found. Using default configuration.")
        config = {
            'base_model': 'efficientnet_b3',
            'unfrozen_blocks': [-1],
            'classifier_layers': [
                ('Linear', (1536, 48)),  # in_features for EfficientNet-B3 is 1536
                ('ReLU', None),
                ('Dropout', 0.5),
                ('Linear', (48, 2))
            ]
        }
    
    # Reconstructing the architecture
    model = models.efficientnet_b3(weights=None)  # We load our own weights
    in_features = model.classifier[1].in_features  


    # Rebuilding same classifier exactly as in training
    classifier_layers = []
    for layer in config['classifier_layers']:
        if layer[0] == 'Linear':
            classifier_layers.append(nn.Linear(*layer[1]))
        elif layer[0] == 'ReLU':
            classifier_layers.append(nn.ReLU())
        elif layer[0] == 'Dropout':
            classifier_layers.append(nn.Dropout(layer[1]))
    
    model.classifier = nn.Sequential(*classifier_layers)
    
    # Replicate freezing/unfreezing
    for param in model.parameters():
        param.requires_grad = False
        
    for idx in config['unfrozen_blocks']:
        for param in model.features[idx].parameters():
            param.requires_grad = True
            
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # Load the saved weights
    try:
        model.load_state_dict(torch.load("best_model_weights.pth", map_location=torch.device('cpu'), weights_only=True))
    except RuntimeError as e:
        st.error(f"Error loading model weights: {e}")
        return None
    
    # Setting model to evaluation mode for prediction
    model.eval()
    return model
model = load_model()
# ----------------------------
# Transforms
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])


# Titre de l'application  ----------------------------
st.title("🔍 Détection de Voiture Tesla avec EfficientNet-B3")
st.markdown("Cette application permet de prédire si une image contient est une voiture Tesla et visualise les performances du modèle.")

# ----------------------------
# Upload d'image
# ----------------------------
st.subheader("Téléchargez votre image ici")
uploaded_file = st.file_uploader("📤 Uploadez une image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Image uploadée", use_container_width=True)

    # Preprocess the image and adding batch dimension
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        # Get model output (raw scores/logits for predict)
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1).squeeze(0)  
        probabilities = probabilities.cpu().numpy() 
        prob_non_tesla = probabilities[0] * 100  
        prob_tesla = probabilities[1] * 100      
        
        # Determining predicted class and confidence
        predicted_class_idx = torch.argmax(output, dim=1).item()  
        predicted_class = "Tesla" if predicted_class_idx == 1 else "Non-Tesla"
        confidence = prob_tesla if predicted_class_idx == 1 else prob_non_tesla


    st.subheader("🔎 Prédiction :")
    if predicted_class == "Tesla":
        st.success(f"🚗 C'est probablement une **Tesla** avec une confiance de {confidence:.2f}%")
    else:
        st.error(f"🚫 Ce n'est probablement **pas** une Tesla avec une confiance de {confidence:.2f}%")
    
    st.subheader("📊 Probabilités des classes :")
    prob_dict = {"Non-Tesla": prob_non_tesla, "Tesla": prob_tesla}
    st.bar_chart(prob_dict)




# ----------------------------
# Graphique des métriques
# ----------------------------

st.header("📈 Performances du modèle")

try:
    with open('metrics_history.pkl', 'rb') as f:
        metrics = pickle.load(f)

    with open('class_names.pkl', 'rb') as f:
        class_names = pickle.load(f)

    epochs = range(1, len(metrics['train_losses']) + 1)

    # Onglets pour chaque graphique
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Loss", "F1 Score", "ROC AUC", "Matrice de Confusion", "Indicateurs de Performance"])

    with tab1:
        st.subheader("📉 Courbes de Loss")
        fig, ax = plt.subplots()
        ax.plot(epochs, metrics['train_losses'], label='Training Loss', color='blue')
        ax.plot(epochs, metrics['val_losses'], label='Validation Loss', color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training vs Validation Loss')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    with tab2:
        st.subheader("📈 F1 Score par Epoch")
        fig_f1, ax_f1 = plt.subplots()
        ax_f1.plot(epochs, metrics['f1_score'], label='F1 Score', color='green', marker='o')
        ax_f1.set_xlabel("Epoch")
        ax_f1.set_ylabel("F1 Score")
        ax_f1.set_title("Évolution du F1 Score")
        ax_f1.grid(True)
        ax_f1.legend()
        st.pyplot(fig_f1)

    with tab3:
        st.subheader("🔺 ROC AUC par Epoch")
        fig_auc, ax_auc = plt.subplots()
        ax_auc.plot(epochs, metrics['roc_auc'], label='ROC AUC', color='red', marker='o')
        ax_auc.set_xlabel("Epoch")
        ax_auc.set_ylabel("ROC AUC")
        ax_auc.set_title("Évolution du ROC AUC")
        ax_auc.grid(True)
        ax_auc.legend()
        st.pyplot(fig_auc)

    with tab4:
        st.subheader("🧮 Matrice de Confusion")
        conf_matrix = metrics['conf_matrices'][-1]
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_xlabel("Classe Prédite")
        ax.set_ylabel("Classe Réelle")
        plt.title("Matrice de Confusion Tesla vs Non-Tesla")
        st.pyplot(fig)

    with tab5:
        st.subheader("📌 Indicateurs de Performaces")
        st.markdown("### Résultats apres 200 epochs") 
        st.write(f"**Train Loss** : {metrics['train_losses'][-1]:.4f}")
        st.write(f"**Validation Loss** : {metrics['val_losses'][-1]:.4f}")
        st.write(f"**F1 Score** : {metrics['f1_score'][-1]:.4f}")
        st.write(f"**Accuracy** : {metrics['accuracy'][-1]:.4f}")
        st.write(f"**ROC AUC** : {metrics['roc_auc'][-1]:.4f}")

except FileNotFoundError:
    st.warning("📉 Fichier `metrics_history.pkl` introuvable. Vérifiez que le fichier est dans le bon répertoire.")
except Exception as e:
    st.warning(f"📉 Erreur lors du chargement ou de l'affichage des métriques : {str(e)}")
