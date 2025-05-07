# open train.json
import json
import os

with open('validation.json', 'r') as f:
    data = json.load(f)
print(len(data))
