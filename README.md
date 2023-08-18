Covid-scans-classification

Overview
This project focuses on the classification of lung scans into two distinct categories:

Normal
1.Covid-19
2.A Convolutional Neural Network (CNN) model is employed to achieve this classification.

Objectives
1.Collect and preprocess lung scan images.
2.Design and train a CNN model using R and the keras library.
3.Evaluate the model's performance on a test dataset.

Data
Lung scan images were sourced from various repositories. The datasets comprise a total of X images (replace X with the actual number), split into training, validation, and testing sets. Note: The dataset is not provided in this repository due to privacy and size considerations.

Methodology
Data Collection: Lung scan images were collected from various sources.
Data Preprocessing:
The images underwent preprocessing tasks such as resizing and normalization.
Data augmentation techniques were applied to enhance the dataset's diversity.

Model Building:
A CNN model with multiple convolutional, pooling, and dense layers was designed.
Specific hyperparameters were chosen based on experimentation and validation performance.
The model was compiled and trained using the training dataset.

#Evaluation:
The performance of the trained model was evaluated on a separate test dataset.
Various metrics and visualizations, such as confusion matrices, were used to assess the model's capabilities.

#Files and Directories
scripts/: Contains the R script (CNN model.R) detailing the data processing, model building, and evaluation steps.
reports/: Includes the presentation (Covid Scans Classification Presentation.pdf) which offers an overview of the project's objectives, methodologies, and results.

#Results
The CNN model achieved commendable accuracy on the test dataset. Detailed results, including visual representations, can be found in the provided presentation.

