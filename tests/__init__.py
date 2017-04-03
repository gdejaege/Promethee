from os import listdir
from os.path import isfile, join

mypath = 'tests/'
files = [f[:-3] for f in listdir(mypath)
         if isfile(join(mypath, f)) and f[-3:] == '.py' and f != '__init__.py']
__all__ = files
