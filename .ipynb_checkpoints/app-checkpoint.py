import streamlit as st

st.title("Plant Disease Classifier")

uploaded_file= st.file_uploader(
    "Upload a leaf image",
    type= ["jpg", "jpeg", "png"]
)

with st.sidebar:
    st.header("About")
    st.write(
        """
        Plant Disease Classifier\n
        Built using MobileNetV2\n
        Validation Accuracy: 95%
        """
    )

# load model    
import tensorflow as tf
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("ft2_model.keras")
model= load_model()

from PIL import Image
import numpy as np

with open("class_names.txt") as f:
        class_names= [line.strip() for line in f]
    
def open_image(uploaded_file):
    img= Image.open(uploaded_file).convert("RGB")
    img= img.resize((224,224))
    img_array= np.array(img)
    img_array= np.expand_dims(img_array, axis=0)
    return img, img_array
    
def pred(img_array):
    result=[]
    confidence=[]
    predictions= model.predict(img_array, verbose=0)[0]
    top3_indices = predictions.argsort()[-3:][::-1]
    for i in top3_indices:
        result.append(class_names[i])
        confidence.append(predictions[i])
    return result, confidence
    
def show_result(result, confidence):
    st.subheader("Top 3 Predictions")
    for i, (cls, cof) in enumerate(zip(result, confidence), start=1):
        st.write(f"{i}. {cls}: {cof:.2%}")

if uploaded_file is not None:
    col1, col2= st.columns(2, gap = "xxlarge")
    with col1:
        st.image(uploaded_file, caption="Uploaded Leaf Image", width=400)
    if st.button("Get results", type= "primary"):
        img, img_array= open_image(uploaded_file)
        with col2:
            with st.spinner("Analyzing leaf..."):
                result, confidence = pred(img_array)
                st.success(f"Detected: {result[0]}")
                st.write(f"Confidence: {confidence[0]:.2%}")
                show_result(result, confidence)
else:
    st.write("Please upload an image.")

