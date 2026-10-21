#python3 -m pip install ai-edge-litert


import tensorflow as tf
import json
import numpy as np
from ai_edge_litert.interpreter import Interpreter

with open("data.json", "r") as f:

    data = json.load(f)
    WORDS = data["words"]
    RESPONSES = data["responses"]

interpreter = Interpreter(model_path = "cataBot.tflite")
interpreter.allocate_tensors()

inIdx = interpreter.get_input_details()[0]["index"]
outIdx = interpreter.get_output_details()[0]["index"]

print("CataBot is active")

while True:
    question = input("\nYou: ").strip().lower()
    if question not in WORDS:
        print("That query is out of my scope")
        continue
    wordID = WORDS[question]
    interpreter.set_tensor(inIdx, np.array([wordID, 0], dtype=interpreter.get_input_details()[0]["dtype"]))
    interpreter.invoke()

    probabilities = interpreter.get_tensor(outIdx)[0]
    winner = str(int(np.argmax(probabilities)))
    
    print(f"Cata: {RESPONSES[winner]}")
