import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import Adam
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

class CustomVGG16(nn.Module):
    def __init__(self):
        super(CustomVGG16, self).__init__()
        self.base_model = models.vgg16(weights='DEFAULT')
        self.base_model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.base_model(x)

# Initialize model, loss function, and optimizer
model = CustomVGG16()
for param in model.base_model.parameters():
    param.requires_grad = False

criterion = nn.BCELoss()
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# Data preprocessing for training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset
train_dataset = datasets.ImageFolder(root=training_images_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop with early stopping
num_epochs = 20
patience = 5
best_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.view(-1), labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # Early stopping logic
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print('Early stopping')
            break

# Load the best model if needed
model.load_state_dict(torch.load('best_model.pth'))