import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging
import matplotlib.pyplot as plt
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class HandwrittenCharDataset(Dataset):
    """
    Custom dataset for handwritten digits and letters.
    """
    def __init__(self, data: pd.DataFrame, transform: Optional[transforms.Compose] = None):
        """
        Args:
            data (pd.DataFrame): DataFrame with 'label' and pixel columns.
            transform (transforms.Compose, optional): Image transformations.
        """
        self.data = data
        self.transform = transform
        self.labels = data['label'].values
        self.images = data.drop('label', axis=1).values.reshape(-1, 28, 28).astype(np.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.images[idx]
        label = self.labels[idx]
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1, 28, 28)
        if self.transform:
            img = self.transform(img)
        return img, label

class CharRecognitionModel(nn.Module):
    """
    CNN model for handwritten character recognition (0-9, A-Z).
    """
    def __init__(self, num_classes: int = 36):
        super(CharRecognitionModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (16, 14, 14)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (32, 7, 7)
            nn.Dropout2d(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

class CharRecognitionPipeline:
    """
    Pipeline for loading, preprocessing, training, and evaluating a character recognition model.
    """
    def __init__(
        self,
        digits_path: str,
        letters_path: str,
        output_dir: str = "results"
    ):
        """
        Args:
            digits_path (str): Path to digits CSV (e.g., MNIST).
            letters_path (str): Path to letters CSV (e.g., NIST).
            output_dir (str): Directory to save outputs.
        """
        self.digits_path = Path(digits_path)
        self.letters_path = Path(letters_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.train_loader = None
        self.test_loader = None

    def load_and_prepare_data(self, samples_per_class: int = 1000) -> None:
        """
        Load and preprocess digits and letters datasets.

        Args:
            samples_per_class (int): Number of samples to select per class.
        """
        logger.info("Loading datasets")
        
        # Load data
        digits_train = pd.read_csv(self.digits_path / "mnist_train.csv")
        digits_test = pd.read_csv(self.digits_path / "mnist_test.csv")
        digits_data = pd.concat([digits_train, digits_test], ignore_index=True)
        letters_data = pd.read_csv(self.letters_path)
        
        # Rename label column
        digits_data.columns = ['label'] + [f'pixel{i}' for i in range(digits_data.shape[1] - 1)]
        letters_data.columns = ['label'] + [f'pixel{i}' for i in range(letters_data.shape[1] - 1)]
        
        # Select samples
        digits_data = digits_data.groupby('label').head(samples_per_class)
        letters_data = letters_data.groupby('label').head(samples_per_class)
        
        # Adjust letter labels (0-25 -> 10-35)
        letters_data['label'] = letters_data['label'] + 10
        
        # Define transforms
        train_transform = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
        ])
        test_transform = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Create datasets
        digits_train, digits_test = train_test_split(
            digits_data, test_size=0.1, stratify=digits_data['label'], random_state=42
        )
        letters_train, letters_test = train_test_split(
            letters_data, test_size=0.1, stratify=letters_data['label'], random_state=42
        )
        
        train_dataset = ConcatDataset([
            HandwrittenCharDataset(digits_train, transform=train_transform),
            HandwrittenCharDataset(letters_train, transform=train_transform)
        ])
        test_dataset = ConcatDataset([
            HandwrittenCharDataset(digits_test, transform=test_transform),
            HandwrittenCharDataset(letters_test, transform=test_transform)
        ])
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, num_workers=2
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=False, num_workers=2
        )
        
        logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    def initialize_model(self) -> None:
        """
        Initialize the CNN model.
        """
        self.model = CharRecognitionModel().to(self.device)
        logger.info("Model initialized")

    def train_model(self, epochs: int = 20, lr: float = 0.001) -> dict:
        """
        Train the model and track metrics.

        Args:
            epochs (int): Number of training epochs.
            lr (float): Learning rate.

        Returns:
            dict: Training and validation metrics history.
        """
        logger.info("Starting training")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        
        for epoch in range(epochs):
            self.model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_loss /= len(self.train_loader)
            train_acc = train_correct / train_total
            
            # Validation
            self.model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in self.test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss /= len(self.test_loader)
            val_acc = val_correct / val_total
            
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
        
        # Save model
        torch.save(self.model.state_dict(), self.output_dir / "char_model.pth")
        return history

    def plot_metrics(self, history: dict) -> None:
        """
        Plot training and validation metrics.

        Args:
            history (dict): Training history with loss and accuracy.
        """
        logger.info("Plotting metrics")
        plt.figure(figsize=(12, 4))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Validation Loss")
        plt.title("Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(history["train_acc"], label="Train Accuracy")
        plt.plot(history["val_acc"], label="Validation Accuracy")
        plt.title("Accuracy Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_metrics.png")
        plt.close()

def main():
    pipeline = CharRecognitionPipeline(
        digits_path="sample_data",  # Adjust to your path
        letters_path="sample_data/A_Z Handwritten Data.csv",
        output_dir="results"
    )
    pipeline.load_and_prepare_data(samples_per_class=1000)
    pipeline.initialize_model()
    history = pipeline.train_model(epochs=20)
    pipeline.plot_metrics(history)

if __name__ == "__main__":
    main()