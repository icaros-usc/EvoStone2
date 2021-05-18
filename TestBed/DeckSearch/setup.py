import glob
import shutil
import os
import subprocess

# Make all the directories that are required for running EvoStone
dirs = ['bin', 'active', 'boxes', 'logs', 'train_log']
for cur_dir in dirs:
    os.makedirs(cur_dir, exist_ok=True)

# First build the entire project
subprocess.call(['dotnet', 'publish', '--configuration', 'Release', '../..'])

ssu_path = '../../SabberStoneUtil/bin/Release/netstandard2.0/publish/*'
ss_path = '../../DeckSearch/bin/Release/netcoreapp3.1/publish/*'
de_path = '../../DeckEvaluator/bin/Release/netcoreapp2.1/publish/*'
model_path = '../../SurrogateModel/bin/Release/netcoreapp3.1/publish/*'
play_path = '../../Playground/bin/Release/netcoreapp3.1/publish/*'
analysis_path = '../../Analysis/bin/Release/netcoreapp3.1/publish/*'
bin_dir = 'bin/'

paths = [ssu_path, ss_path, de_path, model_path, play_path, analysis_path]

for cur_dir in paths:
   for cur_file in glob.glob(cur_dir):
      print(cur_file)
      if not os.path.isdir(cur_file):
         shutil.copy(cur_file, bin_dir)
      else:
         subprocess.call(['cp', '-a', cur_file, bin_dir]) # copy runtimes/ folder to bin

for cur_file in glob.glob('active/*'):
   os.remove(cur_file)
for cur_file in glob.glob('boxes/*'):
   os.remove(cur_file)
# for cur_file in glob.glob('logs/*'):
#    os.remove(cur_file)
