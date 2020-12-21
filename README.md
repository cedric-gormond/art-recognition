> NOTE TO FRENCH READERS :
> Vous trouverez un rapport et synthèse d'évaluation du système à l'adresse suivante : [ici](https://drive.google.com/file/d/1RartA6e97V1tvLfSbB2ithGoaHLQVZl9/view?usp=sharing)

# art-recognition

Machine Learning Art recognition projet based on a BoVW implementation. 

Full implementation made by Cédric Gormond.

# Requirements
- python-3.8.2
- scikit-learn
- numpy
- OpenCV

# BoVW
1. **Preprocessing**
	- Extraction of image descriptors by using BRISK algorithm (based on SIFT)
	- Building codebooks from clusters of image descriptors (Batch Kmeans with a faster convergence). Feature are encoded thanks to vector quantization.
2. **Training**
	- Applying SVC (soft-margin) or KNN on codebooks 

3. **Evaluation**
	- Evaluating accuracy (with confusion matrix) on train and test dataset   

# Dataset
- Kaggle : https://www.kaggle.com/ikarus777/best-artworks-of-all-time
