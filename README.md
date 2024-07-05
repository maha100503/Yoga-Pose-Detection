Dataset=>https://www.kaggle.com/code/aayushmishra1512/yoga-pose-detection/input

YOGA POSE DETECTION:
--Importing Libraries:
imported various Python libraries, including NumPy, pandas, matplotlib, OpenCV, TensorFlow, SHAP, seaborn, and more.
These libraries are essential for data manipulation, visualization, and building machine learning models.
--Data Preparation and Visualization:
You’ve loaded pose estimation images from different categories (labels) using the preprocess_images function.
Each image contains pose landmarks detected by the mp_pose model.
The landmarks’ X and Y coordinates are extracted and stored in a DataFrame (train_images_data).
encoded the labels using LabelEncoder.
The data is organized into features (X coordinates) and labels (categories).
--Model Architecture:
defined a convolutional neural network (CNN) model using TensorFlow/Keras.
The model consists of convolutional layers, max-pooling layers, batch normalization, dropout, and dense layers.
The output layer has softmax activation for multi-class classification.
--Training and Evaluation:
performed 5-fold cross-validation on the training data.
The model’s accuracy and loss are tracked during training.
The mean training and validation accuracies are calculated.
evaluated the model on the test data and obtained the accuracy.
The confusion matrix and classification report provide additional insights.
--Machine Learning Model:
using a RandomForestClassifier with 120 estimators.
The model is trained on the training data (X_train and y_train).
evaluated the model’s accuracy on the testing data (X_test and y_test).
--Accuracy and Confusion Matrix:
The accuracy of your model is calculated using accuracy_score.
The confusion matrix shows how well the model predicts each label (category).

