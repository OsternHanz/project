#import argparse
from typing import Any
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import torch
import numpy as np
from PIL import ImageTk, Image
import os
# 3rdarty
import cv2

# project


def inference_classifier(classifier: Any, path_to_image) -> str:
    image=cv2.imread(path_to_image)
    resized_image = cv2.resize(image, (256, 256))
    normalized_image = resized_image.astype(np.float32) / 255.0
    tensor_image = torch.from_numpy(normalized_image).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        output = classifier(tensor_image)
        print(output)
    _, predicted_idx = torch.max(output, 1)
    class_index = predicted_idx.item()
    class_labels = ["самолет","корабль"]
    predicted_class = class_labels[class_index]
    print(path_to_image)
    img=ImageTk.PhotoImage(Image.open(path_to_image))
    photo=ttk.Label(frame, image=img)
    photo.image=img
    photo.pack()
    text=ttk.Label(frame, text=predicted_class)
    text.pack()




def load_classifier(
    name_of_classifier: str, path_to_pth_weights: str, device: str
) -> Any:
    if device:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model=torch.load(path_to_pth_weights)
    if os.path.isfile(label1["text"]):
        inference_classifier(model, label1["text"])
    else:
        for root, dirs, files in os.walk(label1["text"]):
            for file in files:
                file_path = os.path.join(root, file)
                inference_classifier(model, file_path)



'''def arguments_parser() -> argparse.Namespace:
    """Парсер аргументов

    Returns:
        argparse.Namespace: _description_
    """
    parser = argparse.ArgumentParser(
        description="Скрипт для выполнения классификатора на единичном изображении или папке с изображениями"
    )
    parser.add_argument(
        "--name_of_classifier", "-nc", type=str, help="Название классификатора"
    )
    parser.add_argument(
        "--path_to_weights",
        "-wp",
        type=str,
        help="Путь к PTH-файлу с весами классификатора",
    )
    parser.add_argument(
        "--path_to_content",
        "-cp",
        type=str,
        help="Путь к одиночному изображению/папке с изображениями",
    )
    parser.add_argument(
        "--use_cuda",
        "-uc",
        action="store_true",
        help="Использовать ли CUDA для инференса",
    )
    args = parser.parse_args()

    return args'''


'''def main() -> None:
    """Основная логика работы с классификатором"""
    args = arguments_parser()

    name_of_classifier = args.name_of_classifier
    path_to_weights = args.path_to_weights
    path_to_content = args.path_to_content
    use_cuda = args.use_cuda

    print(f"Name of classifier: {name_of_classifier}")
    print(f"Path to content: {path_to_content}")
    print(f"Path to weights: {path_to_weights}")

    if use_cuda:
        print("Device: CUDA")
    else:
        print("Device: CPU")


if __name__ == "__main__":
    main()'''

def open_file():
    file_path = filedialog.askopenfilename()
    label1["text"]=file_path

def open_file1():
    file_path = filedialog.askopenfilename()
    label2["text"]=file_path

def open_dir():
    file_path = filedialog.askdirectory()
    label1["text"]=file_path

root = tk.Tk()
root.geometry("800x200")
frame = tk.Frame(root)
frame.pack()
label = ttk.Label(frame, text="Модель:")
label.grid(column=0, row=0)
values = ["resnet18", "resnet34", "resnet50"]
combobox = ttk.Combobox(frame, values=values)
combobox.grid(column=1, row=0)
label = ttk.Label(frame, text="Файл/папка с изображением/изображениями:")
label.grid(column=0, row=1)
label1 = ttk.Label(frame, text="Выбрать расположение папки->")
label1.grid(column=1, row=1)
label = ttk.Button(frame, text="Выбор файла", command=open_file)
label.grid(column=2, row=1)
label = ttk.Button(frame, text="Выбор папки", command=open_dir)
label.grid(column=3, row=1)
label = ttk.Label(frame, text="Файл/папка с PTH файлом:")
label.grid(column=0, row=2)
label2 = ttk.Label(frame, text="Выбрать расположение папки->")
label2.grid(column=1, row=2)
label = ttk.Button(frame, text="Выбор файла", command=open_file1)
label.grid(column=2, row=2)
label = ttk.Label(frame, text="Использовать CUDA")
label.grid(column=0, row=3)
checkbox_var = tk.IntVar()
check = ttk.Checkbutton(frame, variable=checkbox_var)
check.grid(column=1, row=3)
label = ttk.Button(frame, text="Готово", command=lambda: load_classifier(combobox.get(), label2["text"], checkbox_var.get()))
label.grid(column=0, row=4)

canvas = tk.Canvas(root)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar = tk.Scrollbar(root, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
canvas.config(yscrollcommand=scrollbar.set)
canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
frame = ttk.Frame(canvas)
canvas.create_window((0, 0), window=frame, anchor="nw")
root.mainloop()
