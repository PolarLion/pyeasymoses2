import os

def get_filename(path):
  files = os.listdir(path)
  if len(files) < 2:
    return ""
  filename = files[0][:-3]
  return filename