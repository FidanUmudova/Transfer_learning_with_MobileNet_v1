# Transfer Learning with MobileNet v2

This repository contains practical implementations of transfer learning using TensorFlow and Keras, focusing on building an efficient image classification model with MobileNetV2.

##  Overview
The goal of this project is to build and fine-tune a binary image classifier using a pre-trained **MobileNetV2** model. Transfer learning allows us to leverage powerful features learned from large datasets (like ImageNet) and adapt them to custom datasets with minimal training time and high accuracy.

##  Features & Workflow
- **Data Augmentation:** Implemented a custom `Sequential` model using `RandomFlip` and `RandomRotation` to increase dataset diversity.
- **Data Preprocessing:** Utilized MobileNetV2's built-in `preprocess_input` function to normalize input images to the `[-1, 1]` range.
- **Feature Extraction:** Frozen the pre-trained base model layers and added custom classification heads (`GlobalAveragePooling2D`, `Dropout`, and `Dense`).
- **Fine-Tuning:** Unfroze specific layers of the base model to fine-tune weights with a reduced learning rate for optimal performance.

##  Requirements
Ensure you have the following libraries installed:
- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib

##  Code Structure
- `data_augmenter()`: Creates data augmentation pipelines.
- `alpaca_model()`: Sets up the MobileNetV2 architecture with a custom classification head.
- **Fine-Tuning Section:** Configures layer unfreezing, loss functions (`BinaryCrossentropy`), and optimizers (`Adam`).

##  Acknowledgments
This project is part of the Convolutional Neural Networks course (Deep Learning Specialization).
