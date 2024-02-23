# # import streamlit as st
# # from keras.models import load_model
# # from keras.preprocessing import image
# # import numpy as np

# # # Load the pre-trained models
# # inception_model = load_model("inceptionv3_model (1) (1).h5")  # Load InceptionV3 model
# # xception_model = load_model("mobilenet_model1 (1).h5")  # Load Xception model
# # mobilenet_model = load_model("mobilenet_model1 (1).h5")  # Load MobileNet model

# # # Function to preprocess input image
# # def preprocess_image(img):
# #     img = image.load_img(img, target_size=(224, 224))
# #     img_array = image.img_to_array(img)
# #     img_array = np.expand_dims(img_array, axis=0)
# #     img_array /= 255.0  # Normalize pixel values
# #     return img_array

# # # Function to predict class label using majority voting
# # def predict_class(img_path):
# #     img = preprocess_image(img_path)
# #     pred1 = inception_model.predict(img)
# #     pred2 = xception_model.predict(img)
# #     pred3 = mobilenet_model.predict(img)
# #     ensemble_pred = np.argmax(pred1 + pred2 + pred3, axis=1)[0]
# #     return ensemble_pred

# # # Streamlit app
# # st.title("Image Classification with Ensemble of InceptionV3, Xception, and MobileNet")

# # uploaded_file = st.file_uploader("Choose an image...", type="jpg")
# # if uploaded_file is not None:
# #     st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
# #     if st.button("Predict"):
# #         prediction = predict_class(uploaded_file)
# #         if prediction == 0:
# #             st.write("Predicted Class: Benign")
# #         else:
# #             st.write("Predicted Class: Malignant")
# import streamlit as st
# from keras.models import load_model
# from keras.preprocessing import image
# import numpy as np

# # Load the pre-trained models
# inception_model = load_model("inceptionv3_model (1) (1).h5")  # Load InceptionV3 model
# xception_model = load_model("mobilenet_model1 (1).h5")  # Load Xception model
# mobilenet_model = load_model("mobilenet_model1 (1).h5")  # Load MobileNet model

# # Function to preprocess input image
# def preprocess_image(img):
#     img = image.load_img(img, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0  # Normalize pixel values
#     return img_array

# # Function to predict class label using majority voting
# def predict_class(img_path):
#     img = preprocess_image(img_path)
#     pred1 = inception_model.predict(img)
#     pred2 = xception_model.predict(img)
#     pred3 = mobilenet_model.predict(img)
    
#     # Aggregate predictions from all models
#     ensemble_pred = np.argmax(pred1 + pred2 + pred3, axis=1)
    
#     # Count occurrences of each class label
#     counts = np.bincount(ensemble_pred)
    
#     # Choose the class label with the highest count
#     predicted_class = np.argmax(counts)
    
#     return predicted_class

# # Streamlit app
# st.title("Image Classification with Ensemble of InceptionV3, Xception, and MobileNet")

# uploaded_file = st.file_uploader("Choose an image...", type="jpg")
# if uploaded_file is not None:
#     st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
#     if st.button("Predict"):
#         prediction = predict_class(uploaded_file)
#         if prediction == 0:
#             st.write("Predicted Class: Benign")
#         else:
#             st.write("Predicted Class: Malignant")
import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Load the trained Inception model
inception_model = load_model("inception_model.h5")

# Function to preprocess input image
def preprocess_image(img):
    img = image.load_img(img, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Function to predict class label using the Inception model
def predict_class(img_path):
    img = preprocess_image(img_path)
    pred = inception_model.predict(img)
    predicted_class = np.argmax(pred, axis=1)[0]
    return predicted_class

# Streamlit app
st.title("Image Classification with Inception Model")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    if st.button("Predict"):
        prediction = predict_class(uploaded_file)
        if prediction == 0:
            st.write("Predicted Class: Class 0")
        else:
            st.write("Predicted Class: Class 1")
