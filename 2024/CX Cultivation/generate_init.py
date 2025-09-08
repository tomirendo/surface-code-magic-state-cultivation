from __future__ import annotations
import stim
import numpy as np
from pyperclip import copy
from tqdm import tqdm
import pymatching

from itertools import zip_longest
from joblib import Parallel, delayed
#%%
import sys; sys.path.insert(0, 'src/')
#%%
import gen
#%%
import pandas as pd
from plotly import express as px
import gen
import matplotlib.pyplot as plt
from json import loads
from sys import argv

with open("generate_d3_init.py") as f:
    d3_generate_file = f.read()
    d1 = int(argv[1])
    d2 = int(argv[2])
    d3_generate_file= d3_generate_file.replace("d = 3", f"d = {d1}")
    d3_generate_file= d3_generate_file.replace("d2 = 15", f"d2 = {d2}")
    exec(d3_generate_file)

