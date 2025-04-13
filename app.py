import os
import torch
from flask import Flask, request, render_template
from torchvision.utils import save_image
import clip
from generator import Generator

app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP model for text encoding.
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# Initialize the Generator with a condition dimension of 512 (CLIP embedding) and noise dimension of 100.
generator = Generator(condition_dim=512, noise_dim=100).to(device)

# Load pre-trained generator weights if available.
if os.path.exists("generator.pth"):
    generator.load_state_dict(torch.load("generator.pth", map_location=device))
generator.eval()

def generate_image_from_text(text):
    """
    Converts a text prompt into an image using a CLIP text encoder and the conditional GAN generator.
    Args:
        text (str): Text prompt provided by the user.
    Returns:
        Tensor: Generated image of shape [1, 3, 64, 64] (assuming a 64x64 output).
    """
    # Tokenize the input text.
    tokens = clip.tokenize([text]).to(device)
    # Obtain the text embedding from CLIP.
    with torch.no_grad():
        text_embedding = clip_model.encode_text(tokens).float()  # Expected shape: [1, 512]
    # Generate a random noise vector.
    noise = torch.randn(1, 100, device=device)
    # Generate the image using the generator.
    with torch.no_grad():
        fake_image = generator(noise, text_embedding)  # Expected output shape: [1, 3, 64, 64]
    return fake_image

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Retrieve the text description from the form.
        text = request.form.get("description")
        # Generate the image based on the text prompt.
        fake_image = generate_image_from_text(text)
        # Save the generated image to the static folder.
        image_path = os.path.join("static", "generated.png")
        save_image(fake_image, image_path)
        # Render the webpage with the generated image and user description.
        return render_template("index.html", image_url=image_path, description=text)
    else:
        # Initial GET request returns a page without an image.
        return render_template("index.html", image_url=None)

if __name__ == "__main__":
    app.run(debug=True)
