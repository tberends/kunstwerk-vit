import gradio as gr
import os
import torch

from model import create_vit_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
class_names = ["gemaal", "stuw"]

# Create EffNetB2 model
vit, vit_transforms = create_vit_model(
    num_classes=len(class_names),
)

# Load saved weights
vit.load_state_dict(
    torch.load(
        f="pretrained_vit_feature_extractor_stuw_gemaal.pth",
        map_location=torch.device("cpu"),
    )
)

# Create predict function
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken."""
    # Start the timer
    start_time = timer()

    # Transform the target image and add a batch dimension
    img = vit_transforms(img).unsqueeze(0)

    # Put model into evaluation mode and turn on inference mode
    vit.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(vit(img), dim=1)

    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {
        class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))
    }

    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)

    # Return the prediction dictionary and prediction time
    return pred_labels_and_probs, pred_time


# Create title, description and article strings
title = "Peilregulerend kunstwerk Vision Transformer 🏰🚧🚀"
description = "Een Vision Transformer feature extractor computer vision model voor het classificeren van kunstwerken in gemalen of stuwen."
article = "Afgeleid van het klassieke voorbeeld om te classificeren of op de afbeelding een kat of een hond staat. Voor het model is het ViT B16 netwerkarchitectuur uit de paper 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale' gebruikt met de gewichten getraind van het model IMAGENET1K_V1 uit de paper 'Training data-efficient image transformers & distillation through attention' getraind op miljoenen afbeeldingen. Dit model is gefinetuned op een dataset van 2000 afbeeldingen van gemalen en stuwen om te bepalen of een afbeelding een gemaal of stuw bevat en heeft een nauwkeurigheid van 90%."

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=2, label="Predictions"),
        gr.Number(label="Prediction time (s)"),
    ],
    examples=example_list,
    title=title,
    description=description,
    article=article,
    theme="Glass",
    live=True,
)

# Launch the demo
demo.launch()