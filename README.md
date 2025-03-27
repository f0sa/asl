![_silent_talkers__ (1)](https://github.com/user-attachments/assets/104bb527-972b-4d32-bacd-29f99e8d8eea)


# asl
A sign language interpreter using live video feed from the camera

## General info
is an AI-powered system designed to bridge the communication gap for individuals with hearing impairments. The project leverages sign language recognition technology to interpret hand gestures and convert them into written text. It supports both Arabic and English sign languages, enabling seamless translation and interaction

# Demo

https://github.com/user-attachments/assets/974b8ae8-8bc5-490a-bdeb-763662e3d248

# screenshot
![Capture](https://github.com/user-attachments/assets/005dbd5b-0a82-48f9-9ca8-b2d91ac40624)



# Technologies and Tools
- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Mediapipe
- TensorFlow

# Setup
Use comand promt to setup environment by using requirements.txt.
``` bash
python3 -m pip rrequirements.txt
```

This will help you in installing all the libraries required for the project

# Process


# Code Examples
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


# ✨ Features:
- ✅ Supports both Arabic and English Sign Language for text conversion.
- ✅ Accurately recognizes hand gestures and converts them into letters, then forms complete sentences.
- ✅ Real-time text generation to facilitate communication.
- ✅ Instant translation between Arabic and English for enhanced accessibility.
- ✅ User-friendly interface designed for ease of use.
- ✅ Scalable and expandable for future improvements and additional language support.


