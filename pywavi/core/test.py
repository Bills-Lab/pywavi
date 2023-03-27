from flanker import Flanker
import pandas as pd
import analysis

test_a = Flanker("/Users/racepeterson/pywavi-2/1010_Flanker_WAVI_eeg.csv", (-100, 500))
analysis.fft_denoise_entire_patient(test_a, "PZ", 300)
