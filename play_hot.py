import subprocess
import os
from glob import glob 

# Path to the virtual environment
venv_path = ".venv"

# Path to the Python executable inside the virtual environment
# Adjust 'bin' to 'Scripts' if you are on Windows
python_executable = os.path.join(venv_path, 'Scripts', 'python.exe')

def process_videos_hot():
  all_videos = glob('./demo/video/*mid.mp4')
  print(all_videos)

  # Loop to run a subprocess 10 times
  for each_video in all_videos:
    
    title = each_video.split(os.sep)[-1]
    print(f'Running HoT for {title}')

    # Define the command to be run
    command = [python_executable, 'demo/vis.py', '--video', title]

    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)
   
    # Print the output
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

if __name__ == '__main__':
  process_videos_hot()