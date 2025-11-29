import json

with open("C:/Users/Mrith/OneDrive/Desktop/ekis-capstone/storage/id_map.json", "r") as f:
    id_map = json.load(f)

print("id_map length =", len(id_map))
print("first few entries =", list(id_map.items())[:3])
