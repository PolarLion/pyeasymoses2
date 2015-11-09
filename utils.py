import os
import time

def write_step (command, easy_config) :
  outfile = open (os.path.join(easy_config.easy_steps, "step.txt"), 'a')
  outfile.write (str (time.strftime('%Y-%m-%d %A %X %Z',time.localtime(time.time())) ) + "\n")
  outfile.write ("pid: " + str(os.getpid()))
  outfile.write (command + "\n")
  outfile.close ()

def get_filename(path):
  files = os.listdir(path)
  if len(files) < 2:
    return ""
  filename = files[0][:-3]
  return filename