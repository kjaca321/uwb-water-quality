import pandas as pd
import numpy as np

salinity_range = (0, 170)
tds_range = (0, 35)
del_temp_range = (-24, 50)
div = 4

data = pd.DataFrame(columns=['salinity (ppt)', 'total dissolved solids (g/L)', 'del temp (deg C)'])

data['salinity (ppt)'] = np.clip(np.random.normal(loc=np.mean(salinity_range), 
                                                  scale=np.mean(salinity_range)/div, size=25000), 
                                                  salinity_range[0], salinity_range[1])
data['total dissolved solids (g/L)'] = np.clip(np.random.normal(loc=np.mean(tds_range), 
                                                  scale=np.mean(tds_range)/div, size=25000), 
                                                  tds_range[0], tds_range[1])
data['del temp (deg C)'] = np.clip(np.random.normal(loc=np.mean(del_temp_range), 
                                                  scale=np.mean(del_temp_range)/div, size=25000), 
                                                  del_temp_range[0], del_temp_range[1])

data.to_csv('../data/eng_water_data.csv')