from simulation import simulator
import pandas as pd

simulator(pd.read_csv('../live/new_data/fil_emp_data.csv'),
          pd.read_csv('../data/eng_water_data.csv')).to_csv(
    '../data/eng_sim_features.csv', index=False
)