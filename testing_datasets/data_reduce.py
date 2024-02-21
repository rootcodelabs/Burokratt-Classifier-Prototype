import json
import random

json_file_path = 'data.json'

with open(json_file_path, 'r') as file:
    data = json.load(file)

if isinstance(data, dict):
    print("JSON data loaded into Python dictionary successfully.")
    data_dict = data
    processed_dict = {}
    for key, value in data_dict.items():
        random_samples = random.sample(value, 50)
        processed_dict[key] = random_samples

    with open("data_r_50", 'w') as file:
        json.dump(processed_dict, file, indent=4)
else:
    print("JSON data does not contain a top-level dictionary.")
