import pandas as pd

def fetch_401K():
    url = 'https://github.com/VC2015/DMLonGitHub/raw/master/sipp1991.dta'
    data = pd.read_stata(url)
    return data

def fetch_bonus():
    url = 'https://raw.githubusercontent.com/VC2015/DMLonGitHub/master/penn_jae.dat'
    data = pd.read_stata(url)
    return data
