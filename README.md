# Music Genre Classification using GTZAN Dataset

A deep learning project to classify music genres using audio features (MFCCs) and spectrogram images, comparing tabular and CNN-based approaches.

## 📌 Project Overview
**Task**: Multi-class classification of songs into 10 genres using the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).  
**Approaches**:
1. **Tabular Data**: Extracted MFCCs (Mel-Frequency Cepstral Coefficients) as features, trained with Scikit-learn (e.g., Random Forest).  
2. **Image Data**: Converted audio to spectrograms, trained a CNN (and ResNet50 via transfer learning).  

**Bonus**: Compared performance between both methods.

## 🛠️ Tools & Libraries  
- **Tabular Modeling**: `scikit-learn` (Random Forest, SVM)  
- **CNN Modeling**: `TensorFlow/Keras`  
- **Transfer Learning**: Pre-trained `ResNet50` on spectrograms  
- **Visualization**: `matplotlib`, `seaborn`
