import sys
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QMessageBox
)
from PyQt5.QtGui import QPainter, QPen, QImage, QPixmap
from PyQt5.QtCore import Qt, QPoint
import numpy as np
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class DrawingCanvas(QWidget):
    """
    A custom widget for drawing with the mouse.
    """
    def __init__(self, parent=None):
        super(DrawingCanvas, self).__init__(parent)
        self.setFixedSize(280, 280)
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.black)
        self.drawing = False
        self.last_point = QPoint()
        self.brush_size = 20

    def paintEvent(self, event):
        """
        Paint the canvas image.
        """
        painter = QPainter(self)
        painter.drawImage(self.rect(), self.image, self.image.rect())

    def mousePressEvent(self, event):
        """
        Start drawing on mouse press.
        """
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        """
        Draw lines as the mouse moves.
        """
        if event.buttons() & Qt.LeftButton and self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.white, self.brush_size, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        """
        Stop drawing on mouse release.
        """
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def clear(self):
        """
        Clear the canvas to black.
        """
        self.image.fill(Qt.black)
        self.update()

    def get_image(self) -> np.ndarray:
        """
        Convert the canvas image to a numpy array for prediction.

        Returns:
            np.ndarray: Grayscale image array of shape (28, 28).
        """
        # Convert QImage to PIL Image
        image = self.image.convertToFormat(QImage.Format_Grayscale8)
        width, height = image.width(), image.height()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(height, width)
        
        # Convert to PIL Image and resize
        pil_image = Image.fromarray(arr).resize((28, 28), Image.LANCZOS)
        return np.array(pil_image)

class HCRApplication(QMainWindow):
    """
    Main application window for handwritten character recognition.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Handwritten Character Recognizer")
        self.setFixedSize(350, 400)
        
        # Initialize model (placeholder; replace with actual model)
        self.model = self.load_model()
        
        # Set up UI
        self.canvas = DrawingCanvas(self)
        self.clear_button = QPushButton("Clear Canvas", self)
        self.predict_button = QPushButton("Predict Character", self)
        
        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.predict_button)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        # Connect buttons
        self.clear_button.clicked.connect(self.canvas.clear)
        self.predict_button.clicked.connect(self.predict)

    def load_model(self) -> torch.nn.Module:
        """
        Load the pre-trained model (placeholder).

        Returns:
            torch.nn.Module: Loaded model.
        """
        # Placeholder: Implement actual model loading
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(784, 36)  # 0-9, A-Z
            def forward(self, x):
                return torch.softmax(self.fc(x), dim=-1)
        
        model = DummyModel()
        model.eval()
        logger.info("Loaded model (placeholder)")
        return model

    def decode_prediction(self, pred_idx: int) -> str:
        """
        Convert prediction index to character.

        Args:
            pred_idx (int): Predicted class index.

        Returns:
            str: Decoded character (0-9 or A-Z).
        """
        if pred_idx < 10:
            return str(pred_idx)
        return chr(65 + (pred_idx - 10))  # 10 -> A, 11 -> B, ..., 35 -> Z

    def predict(self):
        """
        Predict the character drawn on the canvas.
        """
        try:
            # Get image from canvas
            img = self.canvas.get_image()
            img = img.astype(np.float32) / 255.0
            img = torch.tensor(img, dtype=torch.float32).flatten().unsqueeze(0)  # (1, 784)
            
            # Predict
            with torch.no_grad():
                output = self.model(img)
                pred_idx = torch.argmax(output, dim=1).item()
            
            # Show prediction
            char = self.decode_prediction(pred_idx)
            QMessageBox.information(self, "Prediction Result", f"Predicted character: {char}")
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            QMessageBox.critical(self, "Error", "Failed to predict character")

def main():
    app = QApplication(sys.argv)
    window = HCRApplication()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()