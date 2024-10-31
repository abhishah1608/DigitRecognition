# For GUI components (Tkinter Canvas, Button)
from tkinter import Tk, Canvas, Button, LEFT, RIGHT

# For image handling and manipulation
from PIL import Image, ImageDraw, ImageOps

# For numerical operations (e.g., normalizing image array)
import numpy as np

# For loading and using the pre-trained model
import tensorflow as tf

class NumberDrawApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Number")
        self.canvas = Canvas(root, width=200, height=200, bg="white")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)
        
        # Buttons to clear and predict
        self.clear_button = Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=LEFT)
        self.predict_button = Button(root, text="Predict", command=self.predict_digit)
        self.predict_button.pack(side=RIGHT)
        
        self.image = Image.new("L", (200, 200), "white")
        self.draw = ImageDraw.Draw(self.image)
    
    def paint(self, event):
        x, y = event.x, event.y
        r = 8  # Radius of the brush
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill="black")
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (200, 200), "white")
        self.draw = ImageDraw.Draw(self.image)
    
    def predict_digit(self):
        # Preprocess the image for model input
        img = self.image.resize((28, 28))  # Resize to 28x28
        img = ImageOps.invert(img)  # Invert colors: background white, digit black
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for model (batch_size, 28, 28, 1)
        new_model = tf.keras.models.load_model("number_identify_model.h5")
        # Predict using the loaded model
        prediction = new_model.predict(img_array)
        digit = np.argmax(prediction)
        print(f"Predicted digit: {digit}")
