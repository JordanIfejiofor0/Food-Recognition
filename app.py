import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from modelv2 import CNN

model = CNN()
checkpoint = torch.load("v2/foodv2_model.pth", map_location="cpu")

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

class_names = ['Baked Potato',
 'Burger',
 'Crispy Chicken',
 'Donut',
 'Fries',
 'Hot Dog',
 'Pizza',
 'Sandwich',
 'Taco',
 'Taquito']

transform_test = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

st.title("🍔 Food Classifier")

st.markdown("Upload an image of any food and the model will detect what food it is")

uploaded_file = st.file_uploader("Upload an image",type=["png","jpg","jpeg"])

#if we uploaded a file
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image,caption="Uploaded Image",use_container_width=True)

    input_tensor = transform_test(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        _,pred = torch.max(output,1)

        prediction = class_names[pred.item()]

    st.success(f"Predicted Food **{class_names[pred.item()]}**")
        