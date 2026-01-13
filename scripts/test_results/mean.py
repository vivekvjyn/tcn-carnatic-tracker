import json
import os
import numpy as np
import mirdata

data_home = "../../data/"
carn = mirdata.initialize('compmusic_carnatic_rhythm', version='full_dataset_1.0', data_home=data_home)
carn_tracks = carn.load_tracks()
carn_keys = list(carn_tracks.keys())

total = {}
seed = '52'
for json_file in os.listdir(seed):
    json_path = os.path.join(seed, json_file)

    with open(json_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    track_id = result["track_id"]

    track = carn_tracks[track_id]
    taala = track.taala

    if taala not in total.keys():
        total[taala] = {}
        for key in result.keys():
            if key not in total[taala].keys() and key not in ["track_id"]:
                total[taala][key] = [result[key]]
    else:
        for key in result.keys():
            if key == "track_id":
                continue
            total[taala][key].append(result[key])

average = {}
for taala in total.keys():
    average[taala] = {}
    for key in total[taala].keys():
        average[taala][key] = np.mean(total[taala][key])

with open(f"mean_results{seed}.json", "w", encoding="utf-8") as f:
    json.dump(average, f, indent=4)
