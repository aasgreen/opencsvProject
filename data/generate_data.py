import numpy as np
import matplotlib.pyplot as plt
import simplePlot
from pathlib import Path
import json

data_path = Path('.')
#generate name list
names = [data_path / Path('plt'+str(i)) for i in np.arange(100)]

metaData = {str(name): simplePlot.rand_plot(str(name)) for name in names}
with open('metaData.json', 'w') as write_file:
    json.dump(metaData, write_file)
