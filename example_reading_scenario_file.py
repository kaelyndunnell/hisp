from helpers import Scenario

my_scenario = Scenario("scenario.txt")

print(my_scenario.data)

import numpy as np

times = np.linspace(0, my_scenario.get_maximum_time(), 1000)
pulse_types = []
for t in times:
    pulse_type = my_scenario.get_pulse_type(t)
    pulse_types.append(pulse_type)

import matplotlib.pyplot as plt

# color the line based on the pulse type
color = {
    "FP": "red",
    "ICWC": "blue",
    "RISP": "green",
    "GDC": "orange",
    "BAKE": "purple",
}

colors = [color[pulse_type] for pulse_type in pulse_types]

for i in range(len(times) - 1):
    plt.plot(times[i : i + 2], np.ones_like(times[i : i + 2]), c=colors[i])
# plt.xscale("log")
plt.show()
