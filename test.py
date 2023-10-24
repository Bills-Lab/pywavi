from pywavi import WaviDataset
from pywavi.core.preprocessing import butter_lowpass_filter, butter_highpass_filter, apply_notch_filter
from pywavi.util.parse import parse_raw
from deprecated.parse import parse_raw as pr
import matplotlib.pyplot as plt
import numpy as np
import time

a = parse_raw("./dataset")[0].data[:1000,0:21]

# start_time_second = time.time()
# b = pr("./dataset")[0].data[:1000,1:20]
# print(b[0])
# endtime_second = time.time()
# elapsed_time_second = endtime_second - start_time_second

# print(elapsed_time_first, "||", elapsed_time_second)
# a = parse_raw("./dataset")[0].data[:1000,1:20]
# a = a[:,0]
# b = butter_lowpass_filter(a, cutoff=65)
# b = butter_highpass_filter(b, cutoff=0.1)
# b = apply_notch_filter(b)

# fig, ax = plt.subplots()

# # Plot each column of the data as a separate line
# ax.plot(a, label='Raw')
# ax.plot(b, label='Filtered')

# # Add a legend and axis labels
# ax.legend()
# ax.set_xlabel('X')
# ax.set_ylabel('Y')

# # Show the plot
# plt.show()