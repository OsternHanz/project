from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import os
import cv2

def open_file():
    file_path = filedialog.askopenfilename()
    label_path["text"]=file_path

def classifier():
    path=label_path["text"]
    result=model(path)
    image=cv2.imread(path)
    for res in result:
        boxes=res.boxes
        for box in boxes:
            object_name = model.names[int(box.cls)]
            cords=box.xyxy
            x, y, x1, y1=int(cords[0][0]), int(cords[0][1]), int(cords[0][2]), int(cords[0][3])
            cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 2)
            cv2.putText(image, object_name, (x, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Image with Rectangle and Label", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




model=YOLO(r"runs\detect\train2\weights\best.pt")
root=tk.Tk()
root.geometry("500x500")
label_name=ttk.Label(text="Выберите картинку: ")
label_name.grid(row=0, column=0)
label_path=ttk.Label(text="Здесь будет ваш путь")
label_path.grid(row=0, column=1)
button_photo=ttk.Button(text="Выбрать", command=open_file)
button_photo.grid(row=0, column=2)
button_ready=ttk.Button(text="Готово", command=classifier)
button_ready.grid(row=1, column=0)
root.mainloop()
