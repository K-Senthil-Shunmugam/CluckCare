# CluckCare
🐔🐥🐓 CluckCare is a website harnessing the power of deep learning convolutional neural networks (CNN) to predict chicken diseases from uploaded images of their excretions. Our simple yet effective approach aims to assist poultry farmers and veterinarians in promptly identifying potential health issues.


# CNN Model for Image Classification

This repository contains a Convolutional Neural Network (CNN) model implemented in TensorFlow/Keras for image classification. The model is trained on a dataset consisting of images belonging to three different classes.

## Requirements

- TensorFlow
- scikit-learn
- OpenCV
- NumPy
- Pandas
- Matplotlib

## Dataset

The dataset used for training and testing the model is located in the `DataSet` directory. It consists of images categorized into three classes. The dataset is split into training and testing sets using the `train_test_split` function from scikit-learn.

## Model Architecture

The CNN model architecture is based on the VGG architecture. It consists of several convolutional layers followed by max-pooling layers. The final layers include fully connected layers with ReLU activation and a softmax output layer for classification.

## Model Training

The model is trained using the training data generated using `ImageDataGenerator` from Keras. The training process is visualized using Matplotlib. The model is compiled using the Adam optimizer and categorical cross-entropy loss function.

## Model Evaluation

The model achieves an accuracy of approximately 96.51% on the validation set. Model evaluation metrics include loss and accuracy plotted over epochs.

## Usage

To use this model, follow these steps:

1. Install the required dependencies listed in the `Requirements` section.
2. Clone this repository.
3. Navigate to the repository directory.
4. Run the training script.

```bash
python train_model.py
