## **👶 Beginner Exercise: Generate Random Noise (The Starting Point of GANs)**
GANs start with **random noise** before generating realistic images. Let’s create a **random noise vector** that a GAN could use.  

### **Task:**  
- Generate **a batch of random noise** from a normal distribution (mean=0, std=1).  
- Use NumPy or PyTorch to generate **100 random vectors**, each of size **(1, 100)**.  

### **Code Template (Python + NumPy)**
```python
import numpy as np

# Generate 100 random noise vectors of size (1, 100)
random_noise = np.random.normal(0, 1, (100, 100))

# Print shape to confirm
print("Random noise shape:", random_noise.shape)
```
✅ **Expected Output:**  
```
Random noise shape: (100, 100)
```
🎯 **Goal:** Understand how GANs start with randomness.

---

## **🏗 Intermediate Exercise: Build a Simple Generator (Fake Image Maker)**
A **Generator** takes random noise and tries to create an image. Let’s build a simple one using **PyTorch**.

### **Task:**  
- Create a simple **Generator** model using **PyTorch**.
- Use a **fully connected neural network** that **maps a random vector to an image**.
- The output should be a **28×28 image (like MNIST digits).**

### **Code Template (PyTorch)**
```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),  # Input size 100 (random noise)
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),  # Output image size (flattened)
            nn.Tanh()  # Normalize values between -1 and 1
        )

    def forward(self, x):
        return self.model(x)

# Create a Generator instance
gen = Generator()

# Generate random noise
random_noise = torch.randn(1, 100)  # One random noise vector

# Generate a fake image
fake_image = gen(random_noise)

# Print shape
print("Fake image shape:", fake_image.shape)
```
✅ **Expected Output:**  
```
Fake image shape: torch.Size([1, 784])
```
🎯 **Goal:** Learn how a Generator turns noise into an image.

---

## **🔥 Advanced Exercise: Train a Mini GAN (Generator + Discriminator)**
Now, let’s train **a tiny GAN** on MNIST (handwritten digits). This is a **simplified GAN** training loop.

### **Task:**  
- Create **both** Generator and Discriminator.  
- Train them with the **MNIST dataset** using **PyTorch**.  
- The Discriminator should classify images as **real or fake**.  
- The Generator should try to **fool the Discriminator**.  

### **Code Template (GAN Training on MNIST)**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Generator Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Initialize models
gen = Generator()
disc = Discriminator()

# Loss function & Optimizers
loss_function = nn.BCELoss()
optimizer_gen = optim.Adam(gen.parameters(), lr=0.0002)
optimizer_disc = optim.Adam(disc.parameters(), lr=0.0002)

# Training Loop
for epoch in range(5):  # 5 epochs for quick training
    for real_images, _ in dataloader:
        # Flatten real images
        real_images = real_images.view(-1, 28*28)

        # Create labels
        real_labels = torch.ones(real_images.size(0), 1)  # Real images = 1
        fake_labels = torch.zeros(real_images.size(0), 1)  # Fake images = 0

        # Train Discriminator
        optimizer_disc.zero_grad()
        real_predictions = disc(real_images)
        real_loss = loss_function(real_predictions, real_labels)

        # Generate fake images
        random_noise = torch.randn(real_images.size(0), 100)
        fake_images = gen(random_noise)
        fake_predictions = disc(fake_images.detach())  # Detach so Generator isn't updated here
        fake_loss = loss_function(fake_predictions, fake_labels)

        # Total Discriminator loss
        disc_loss = real_loss + fake_loss
        disc_loss.backward()
        optimizer_disc.step()

        # Train Generator
        optimizer_gen.zero_grad()
        fake_predictions = disc(fake_images)  # Now check fake images
        gen_loss = loss_function(fake_predictions, real_labels)  # Try to "trick" discriminator
        gen_loss.backward()
        optimizer_gen.step()

    print(f"Epoch [{epoch+1}/5]: Discriminator Loss = {disc_loss.item():.4f}, Generator Loss = {gen_loss.item():.4f}")
```

✅ **Expected Output (Example after each epoch)**  
```
Epoch [1/5]: Discriminator Loss = 1.1254, Generator Loss = 0.8432
Epoch [2/5]: Discriminator Loss = 0.9218, Generator Loss = 1.1023
...
```
🎯 **Goal:** Train a basic GAN and generate fake handwritten digits!

---

## **🚀 Bonus Challenge: Visualize GAN Images**
- Modify the advanced exercise to **save and display the generated images** every few epochs.
- Use **Matplotlib** to visualize fake images and see how they improve over time.

```python
import matplotlib.pyplot as plt

# Generate some fake images
random_noise = torch.randn(16, 100)  # Generate 16 images
fake_images = gen(random_noise).detach().numpy()  # Convert to NumPy

# Reshape and plot
fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(fake_images[i].reshape(28, 28), cmap='gray')
    ax.axis('off')

plt.show()
```

---

## **🎯 Summary of What You Learned**
✅ **Beginner:** Generate random noise vectors (GANs start with randomness).  
✅ **Intermediate:** Build a simple Generator to turn noise into images.  
✅ **Advanced:** Train a full **mini-GAN** to generate fake MNIST digits!  
