from tkinter import *
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier

# ------------------ GUI ------------------
main = tk.Tk()
main.title("Crop Prediction using Machine Learning")
main.geometry("1000x650")

# ------------------ Globals ------------------
filename = None
cls = None
X_train = X_test = y_train = y_test = None
dct_acc = random_acc = pac_acc = 0

# ------------------ Functions ------------------
def upload():
    global filename
    filename = filedialog.askopenfilename()
    text.delete('1.0', END)
    text.insert(END, filename + " loaded successfully\n")

def generateModel():
    global X_train, X_test, y_train, y_test

    data = pd.read_csv(filename)

    X = data.iloc[:, 0:7].values
    y = data.iloc[:, 7].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    text.insert(END, "Train-Test Split Completed\n")
    text.insert(END, f"Total Samples: {len(data)}\n")
    text.insert(END, f"Training Samples: {len(X_train)}\n")
    text.insert(END, f"Testing Samples: {len(X_test)}\n\n")

def prediction(X_test, model):
    return model.predict(X_test)

def cal_accuracy(y_test, y_pred, name):
    acc = accuracy_score(y_test, y_pred) * 100
    text.insert(END, f"{name} Accuracy: {acc:.2f}%\n\n")
    return acc

def runDCT():
    global cls, dct_acc
    cls = DecisionTreeClassifier()
    cls.fit(X_train, y_train)

    y_pred = prediction(X_test, cls)
    dct_acc = cal_accuracy(y_test, y_pred, "Decision Tree")

def runRF():
    global cls, random_acc
    cls = RandomForestClassifier(n_estimators=50, random_state=0)
    cls.fit(X_train, y_train)

    y_pred = prediction(X_test, cls)
    random_acc = cal_accuracy(y_test, y_pred, "Random Forest")

def runPAC():
    global cls, pac_acc
    cls = PassiveAggressiveClassifier()
    cls.fit(X_train, y_train)

    y_pred = prediction(X_test, cls)
    pac_acc = cal_accuracy(y_test, y_pred, "Passive Aggressive")

def predicts():
    if cls is None:
        text.insert(END, "❌ Train a model first!\n")
        return

    testfile = filedialog.askopenfilename()
    testdata = pd.read_csv(testfile).iloc[:, 0:7].values
    y_pred = cls.predict(testdata)

    crops = {
        1: "Rice",
        2: "Wheat",
        3: "Mung Bean",
        4: "Tea",
        5: "Millet",
        6: "Maize",
        7: "Lentil"
    }

    for i in range(len(testdata)):
        text.insert(END, f"Input: {testdata[i]} → Crop: {crops.get(y_pred[i],'Unknown')}\n")

def graph():
    if dct_acc == 0 and random_acc == 0 and pac_acc == 0:
        text.insert(END, "❌ Run models first\n")
        return

    models = ['Decision Tree', 'Random Forest', 'PAC']
    acc = [dct_acc, random_acc, pac_acc]

    plt.bar(models, acc)
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison")
    plt.show()

# ------------------ UI ------------------
font = ('times', 14, 'bold')

Button(main, text="Upload Dataset", command=upload).place(x=10, y=100)
Button(main, text="Preprocess", command=generateModel).place(x=250, y=100)
Button(main, text="Decision Tree", command=runDCT).place(x=420, y=100)
Button(main, text="Random Forest", command=runRF).place(x=620, y=100)
Button(main, text="Passive Aggressive", command=runPAC).place(x=10, y=150)
Button(main, text="Accuracy Graph", command=graph).place(x=250, y=150)
Button(main, text="Predict Crop", command=predicts).place(x=450, y=150)

text = Text(main, height=20, width=120)
text.place(x=10, y=250)

main.mainloop()
