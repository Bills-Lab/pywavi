from flanker import Flanker
import pandas as pd

test_a = Flanker("/Users/racepeterson/pywavi-2/1010_Flanker_WAVI_eeg.csv", (-100, 500))
print(test_a[0].correct)
print(test_a[0].reaction)
