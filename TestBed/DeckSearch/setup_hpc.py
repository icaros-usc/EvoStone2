import glob
import os

# Make all the directories that are required for running EvoStone
dirs = ['bin', 'active', 'boxes', 'logs', 'train_log']
for cur_dir in dirs:
    os.makedirs(cur_dir, exist_ok=True)

for cur_file in glob.glob('active/*'):
   os.remove(cur_file)
for cur_file in glob.glob('boxes/*'):
   os.remove(cur_file)
# for cur_file in glob.glob('logs/*'):
#    os.remove(cur_file)
