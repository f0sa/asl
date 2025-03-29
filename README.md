![_silent_talkers__ (1)](https://github.com/user-attachments/assets/104bb527-972b-4d32-bacd-29f99e8d8eea)


# asl
A sign language interpreter using live video feed from the camera

## General info
is an AI-powered system designed to bridge the communication gap for individuals with hearing impairments. The project leverages sign language recognition technology to interpret hand gestures and convert them into written text. It supports both Arabic and English sign languages, enabling seamless translation and interaction


## Table of contents
* [Demo](#demo)
* [Screenshots](#screenshot)
* [Technologies and Tools](#technologies-and-tools)
* [Requirements](#requirements)
* [Setup](#setup)
* [Code Examples](#code-examples)
* [Features](#features)
* [Future Improvements](future-improvements)
* [License](#license)

## Demo

https://github.com/user-attachments/assets/974b8ae8-8bc5-490a-bdeb-763662e3d248

- You can also try it through the link: 
https://huggingface.co/spaces/ahmedos13/SilenTalker

## screenshot
![Capture](https://github.com/user-attachments/assets/005dbd5b-0a82-48f9-9ca8-b2d91ac40624)
![image](https://github.com/user-attachments/assets/eebe6794-9ad8-4e56-99e0-4a3356ebeadf)



## Technologies and Tools
- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Mediapipe
- TensorFlow

## Requirements

Use comand promt to setup environment by using requirements.txt.
``` bash
python3 -m pip requirements.txt
```

This will help you in installing all the libraries required for the project

## Setup
how to run it for the first time >>

- Data Preparation
Store ASL gesture images in the Train/ and Test/ directories.
Ensure images are categorized into subdirectories named after each corresponding letter.

- Train the Model

Run the following script to train the model:
```bash
python train_model.py
```

This script:
Preprocesses images.
Trains the model using EfficientNetB3.

Saves the trained model as translator.keras.

- Evaluate the Model

Run the following command to test the model's accuracy on the test dataset:
```bash
python evaluate_model.py
```

- Real-Time ASL Translation
To run real-time ASL recognition using the webcam:
```bash
python real_time_translator.py
```
This script:
- Captures hand gestures using OpenCV.
- Predicts the corresponding letter.
- Displays the prediction in real-time.

## Code Examples
```bash

# ======================
# Training Setup
# ======================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
    ModelCheckpoint('English_translator.keras', save_best_only=True)
]

# Initial training (feature extraction phase)
history = model.fit(
    train_gen,
    epochs=15,  # Train for 15 epochs before fine-tuning
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)

```


## Features:
- ✅ Supports both Arabic and English Sign Language for text conversion.
- ✅ Accurately recognizes hand gestures and converts them into letters, then forms complete sentences.
- ✅ Real-time text generation to facilitate communication.
- ✅ Instant translation between Arabic and English for enhanced accessibility.
- ✅ User-friendly interface designed for ease of use.
- ✅ Scalable and expandable for future improvements and additional language support.

## Future Improvements

- Expand dataset for better accuracy.
- Support full-word translation instead of letter-by-letter.
- Improve real-time processing speed.

## License

This project is open-source and free to use.

