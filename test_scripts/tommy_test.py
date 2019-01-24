# tommy_test.py
import numpy as np
import os

def set_base_dir():
    """ SET path directory to the repo base """
    os.chdir(os.path.abspath(
      os.path.join(
        os.path.dirname( __file__ ),'..'
        )
      )
    )
    print(("Working from dir:", os.getcwd()))

    return

result = np.load('6 result_analysis/paper_result.npy')
corr = np.load('6 result_analysis/corr.npy')

# 24 'results' of different metric:algorithm permutations
compare_result_final = np.load('6 result_analysis/Compare_result_final.npz')
compare_result_ridge = np.load('6 result_analysis/Compare_result_ridge.npz')
compare_result = np.load('6 result_analysis/Compare_result.npz')

# read locations file
df = pd.read_csv('data/locations_final.csv', names=["state_id","county_id","lat","lon"])
# order by state_id / county_id
df = df.sort_values(['state_id','county_id'])
# save to new csv file
df[df.state_id == 1].to_csv('data/subset_locations.csv', header=False, index=False)
