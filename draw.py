import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from network import forward_pass

W1 = np.load('W1.npy')
W2 = np.load('W2.npy')
b1 = np.load('b1.npy')
b2 = np.load('b2.npy')

canvas_size = 280

window = tk.Tk()
window.title("Draw a Digit")

canvas = tk.Canvas(window, width=canvas_size, height=canvas_size, bg='black')
canvas.pack()

image = Image.new('L', (canvas_size, canvas_size), color=0)
draw = ImageDraw.Draw(image)

def paint(event):
    x, y = event.x, event.y
    r = 10
    canvas.create_oval(x-r, y-r, x+r, y+r, fill='white', outline='white')
    draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

def predict():
    resized = image.resize((28, 28), Image.LANCZOS)
    img_array = np.array(resized) / 255.0
    output = forward_pass(img_array, W1, W2, b1, b2)[0]
    digit = np.argmax(output)
    confidence = np.max(output) * 100
    result_label.config(text=f"Prediction: {digit}  Confidence: {confidence:.1f}%")

def clear():
    canvas.delete('all')
    draw.rectangle([0, 0, canvas_size, canvas_size], fill=0)
    result_label.config(text="Prediction: ")

canvas.bind('<B1-Motion>', paint)

btn_frame = tk.Frame(window)
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="Predict", command=predict, width=10).pack(side='left', padx=5)
tk.Button(btn_frame, text="Clear", command=clear, width=10).pack(side='left', padx=5)

result_label = tk.Label(window, text="Prediction: ", font=('Arial', 16))
result_label.pack()

window.mainloop()