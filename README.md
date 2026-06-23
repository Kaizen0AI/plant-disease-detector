# Plant Disease Classifier Web App

A web application built with Streamlit and TensorFlow that identifies plant diseases from leaf images using a fine-tuned MobileNetV2 model.

## Features

* Upload a leaf image (`.jpg`, `.jpeg`, `.png`)
* Predict plant disease using a fine-tuned MobileNetV2 model
* Display Top 3 predictions with confidence scores
* User-friendly web interface built with Streamlit

## Model Information

* Architecture: MobileNetV2
* Transfer Learning from ImageNet
* Fine-Tuning on PlantVillage Dataset
* Validation Accuracy: **95.0%**
* Number of Classes: **15**

## Dataset

* PlantVillage Dataset
* 15 disease/healthy plant classes
* Images resized to 224×224 RGB

## Screenshot

*Add a screenshot of the application here.*

## Installation

Clone the repository:

```bash
git clone https://github.com/Kaizen0AI/plant-disease-streamlit-app.git
cd plant-disease-streamlit-app
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
streamlit run app.py
```

The application will open automatically in your browser.

## Project Structure

```text
plant-disease-streamlit-app/
│
├── app.py
├── ft2_model.keras
├── class_names.txt
├── requirements.txt
└── README.md
```

## How It Works

1. Upload a leaf image.
2. The image is resized to 224×224 pixels.
3. The fine-tuned MobileNetV2 model generates predictions.
4. The application displays the Top 3 predicted classes along with confidence scores.

## Future Improvements

* Deploy online using Streamlit Community Cloud
* Add confidence visualizations
* Support more plant species and diseases
* Improve UI and user experience

## Author

Built by MD. Kamil as part of a deep learning and computer vision learning journey.

