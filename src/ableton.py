import live
import random
from pythonosc import udp_client
import time

params_to_randomize = [4,39,40,41]
param_ranges = [
    [0.0, 1.0],
    [0.001, 0.5],
    [0.001, 0.2],
    [0.01, 0.6]
]
def randomize(file,device):
    for idx, param in enumerate(params_to_randomize):
        parameter = device.parameters[param]
        parameter.value = random.uniform(param_ranges[idx][0], param_ranges[idx][1])
        file.write(f"{str(parameter.value[3])}, ")
    file.write("\n")


set = live.Set(scan=True)
set.tempo = 120.0
track = set.tracks[0]
print("Track name '%s'" % track.name)

ip = "127.0.0.1"
port = 11000
client = udp_client.SimpleUDPClient(ip, port)

track_index = 1
clip_slot_index = 0
client.send_message(f"/live/clip_slot/delete_clip", [track_index, clip_slot_index])

wavetable = track.devices[0]

with open("parameters.dat", "w") as file:
    for i in range(20):
        randomize(file, wavetable)
        time.sleep(0.3)
        client.send_message(f"/live/clip_slot/fire", [track_index, clip_slot_index])
        time.sleep(2.3)
        set.stop_playing()
        time.sleep(0.1)
        client.send_message(f"/live/clip_slot/delete_clip", [track_index, clip_slot_index])


# for i in range(100):
#     print(wavetable.parameters[i].get_value(), wavetable.parameters[i], i)
