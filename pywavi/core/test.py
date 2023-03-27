from flanker import Flanker
import pandas as pd
import analysis
from key import ALL_REGIONS
# print(len(ALL_REGIONS))
test_a = Flanker("/Users/racepeterson/pywavi-2/1010_Flanker_WAVI_eeg.csv", (-100, 500))
# analysis.fft_denoise_entire_patient(test_a, "PZ", 300)
analysis.fft_denoise_average_of_patient(test_a, "PZ", 300)
