import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import clip
from generator import Generator

def truncate_prompt(prompt, max_words=20):
    """
    Truncates the prompt to a maximum number of words.
    Adjust max_words if needed to stay within CLIP's context length.
    """
    words = prompt.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words])
    else:
        return prompt

def find_image_filename(base_name, folder, extensions=None):
    """
    Given a base filename (without extension), try to find a file in 'folder'
    by appending one of the allowed extensions.
    """
    if extensions is None:
        extensions = ['.jpg', '.png', '.jpeg']
    base_name = base_name.strip()
    if os.path.splitext(base_name)[1]:
        if os.path.exists(os.path.join(folder, base_name)):
            return base_name
    for ext in extensions:
        candidate = base_name + ext
        if os.path.exists(os.path.join(folder, candidate)):
            return candidate
    return None

##############################################
# 1. Custom Dataset with Precomputed Embeddings
##############################################
class FoodRecipeDataset(Dataset):
    def __init__(self, csv_file, images_folder, transform=None, device="cpu"):
        """
        Args:
            csv_file (string): Path to the CSV file.
            images_folder (string): Folder with all the images.
            transform (callable, optional): Transform to be applied on an image.
            device (string): "cuda" or "cpu"
        """
        # Read CSV file
        self.data = pd.read_csv(csv_file)
        self.images_folder = images_folder
        self.transform = transform
        self.device = device

        print(f"Initial CSV rows: {len(self.data)}")

        # Check that the required column exists.
        if "Image_Name" not in self.data.columns:
            raise ValueError("CSV is missing the required 'Image_Name' column.")

        # Clean up filenames.
        self.data["Image_Name"] = self.data["Image_Name"].astype(str).apply(lambda x: x.strip())

        # For each row, try to find the valid filename (adding extension if needed).
        valid_filenames = []
        for idx, row in self.data.iterrows():
            base_name = row["Image_Name"]
            valid_filename = find_image_filename(base_name, images_folder)
            valid_filenames.append(valid_filename)
        self.data["Valid_File"] = valid_filenames

        # Filter rows where no valid file was found.
        self.data = self.data[self.data["Valid_File"].notnull()].reset_index(drop=True)
        print(f"Valid samples after filtering: {len(self.data)}")

        # Load CLIP model for precomputing text embeddings.
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()

        # Precompute text embeddings.
        self.embeddings = []
        for idx, row in self.data.iterrows():
            # Combine Title and Cleaned_Ingredients into a prompt.
            prompt = f"{row['Title']}: {row['Cleaned_Ingredients']}"
            # Truncate the prompt to avoid exceeding CLIP's context length.
            prompt = truncate_prompt(prompt, max_words=20)
            tokens = clip.tokenize([prompt]).to(device)
            with torch.no_grad():
                emb = self.clip_model.encode_text(tokens).float()  # Expected shape: [1, 512]
            self.embeddings.append(emb.squeeze(0))  # Now shape: [512]

        # Debug: print details of first few samples.
        for idx in range(min(3, len(self.data))):
            print(f"Sample {idx}:")
            print(f"  File: {self.data.iloc[idx]['Valid_File']}")
            print(f"  Prompt: {self.data.iloc[idx]['Title']} | {self.data.iloc[idx]['Cleaned_Ingredients']}")
            print(f"  Truncated Prompt: {truncate_prompt(f'{self.data.iloc[idx]['Title']}: {self.data.iloc[idx]['Cleaned_Ingredients']}', max_words=20)}")
            print(f"  Embedding shape: {self.embeddings[idx].shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_file = row["Valid_File"]
        img_path = os.path.join(self.images_folder, img_file)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        text_embedding = self.embeddings[idx]
        return image, text_embedding

##############################################
# 2. Define the Conditional Discriminator
##############################################
class Discriminator(nn.Module):
    def __init__(self, condition_dim, img_channels=3, feature_map_size=64):
        super(Discriminator, self).__init__()
        self.condition_dim = condition_dim

        self.img_net = nn.Sequential(
            nn.Conv2d(img_channels, feature_map_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_map_size, feature_map_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_map_size * 2, feature_map_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_map_size * 4, feature_map_size * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(feature_map_size * 8 * 4 * 4 + condition_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, img, condition):
        batch_size = img.size(0)
        features = self.img_net(img)
        features = features.view(batch_size, -1)
        combined = torch.cat((features, condition), dim=1)
        validity = self.fc(combined)
        return validity

##############################################
# 3. Hyperparameters and Data Setup
##############################################
batch_size = 64
noise_dim = 100
condition_dim = 512  # Dimension of CLIP text embeddings.
lr = 0.0002
num_epochs = 200
image_size = 64

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Update these paths to match your dataset locations.
csv_path = "C:/Users/ragul/Downloads/Emerging Final Project/Data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
images_folder = "C:/Users/ragul/Downloads/Emerging Final Project/Data/Food Images/"

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = FoodRecipeDataset(csv_file=csv_path, images_folder=images_folder, transform=transform, device=device)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

##############################################
# 4. Initialize Models and Optimizers
##############################################
generator = Generator(condition_dim=condition_dim, noise_dim=noise_dim).to(device)
discriminator = Discriminator(condition_dim=condition_dim).to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

##############################################
# 5. Training Loop
##############################################
for epoch in range(num_epochs):
    for i, (real_images, text_conditions) in enumerate(dataloader):
        batch_size_curr = real_images.size(0)
        real_images = real_images.to(device)
        text_conditions = text_conditions.to(device)
        
        real_labels = torch.ones(batch_size_curr, 1, device=device)
        fake_labels = torch.zeros(batch_size_curr, 1, device=device)
        
        ##########################
        # Train Discriminator
        ##########################
        optimizer_D.zero_grad()
        outputs_real = discriminator(real_images, text_conditions)
        d_loss_real = criterion(outputs_real, real_labels)
        
        noise = torch.randn(batch_size_curr, noise_dim, device=device)
        fake_images = generator(noise, text_conditions)
        outputs_fake = discriminator(fake_images.detach(), text_conditions)
        d_loss_fake = criterion(outputs_fake, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()
        
        ##########################
        # Train Generator
        ##########################
        optimizer_G.zero_grad()
        outputs = discriminator(fake_images, text_conditions)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()
        
        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch {i}/{len(dataloader)} "
                  f"Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}")
    
    # Save checkpoints after each epoch.
    torch.save(generator.state_dict(), "generator.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")
    
    noise_sample = torch.randn(1, noise_dim, device=device)
    sample_condition = text_conditions[0].unsqueeze(0)
    sample_image = generator(noise_sample, sample_condition)
    save_image(sample_image, f"sample_epoch_{epoch+1}.png")

print("Training completed!")
