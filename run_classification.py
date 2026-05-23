import os, sys, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# Change dir to hotel_enterprise just in case
os.chdir('/Users/nabindev/Downloads/hotel_enterprise')

from src.train_models_ts import train_cancellation_model, generate_results_md, BASE, D

print('\n[2/3] Cancellation Classification …')
bookings = pd.read_csv(D('bookings.csv'))
ml_res = train_cancellation_model(bookings)

# Mock ts_results since we already know the MAPEs from the logs
ts_results = {
    'occupancy': {'mape': 0.1505, 'nbeats_mape': 0.1595},
    'adr': {'mape': 0.0739, 'nbeats_mape': float('nan')},
    'revenue': {'mape': 0.2210, 'nbeats_mape': float('nan')}
}

print('\n[3/3] Generating Results Report …')
generate_results_md(ts_results, ml_res)
print('Done!')
