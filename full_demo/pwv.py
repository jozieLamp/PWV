import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from statistics import *
import heartpy as hp
import os

def filterWave(data, sample_rate=240.0, bpmin=0, bpmax=550, lowpass=True, highpass=True, verbose=False):

  # Change values ouside possible range to min and max pulse value
  data = [bpmin if i <= bpmin else (bpmax if i > bpmax else i) for i in data]

  data = hp.filter_signal(data, cutoff=15, sample_rate=sample_rate, order=4, filtertype='lowpass') if lowpass else data
  data = hp.filter_signal(data, cutoff=.01, sample_rate=sample_rate, order=4, filtertype='highpass') if highpass else data
  wd, m = hp.process(hrdata=data, sample_rate=sample_rate, bpmmin=bpmin, bpmmax=bpmax)
  return wd, m

def segmentWave(peakList, sample_rate=240.0):

  rngs = []
  try:
    peakList.insert(0, 0)
  except:
    peakList = np.insert(peakList, 0, 0)
  
  for i in range(len(peakList)-2):
    a = (peakList[i] + peakList[i+1])/2
    b = (peakList[i+1] + peakList[i+2])/2
    interval = .15*(b-a)
    a += interval
    b -= interval
    rngs.append(round(a))
  return rngs

def preprocess(dataDir):

  path = dataDir
  list_of_files = []

  for root, _, files in os.walk(path):
    for file in files:
      list_of_files.append(os.path.join(root,file))
  list_of_files.sort()
  
    
  waveformData = [] # waveform data
  segmentIndices = [] # segmentation indices

  for i in range(len(list_of_files)):

    data = hp.get_data(list_of_files[i], delim = ' ', column_name = 'AO')
    wd, _ = filterWave(data)
    waveformData.append(wd['hr'])
    segmentIndices.append(segmentWave(wd['peaklist']))
  return waveformData, segmentIndices

def calcStats(wave):

  #calculate metrics for each indexed wave saved in individual metric arrays
  #remove outliers, save only mean for each metric
  #append metric means to new df

  #finding maximum, defining region for dic notch
  wave = pd.Series(wave)
  sysMaxi = wave.idxmax()
  reg = wave[int(round(sysMaxi*1.05)):int(round(sysMaxi+(len(wave)*.28)))]
  diff = reg.diff()

  NP = [0,0]


  #find dic notch
  ser = [0,0]
  counter = 0
  for index,value in diff.items():
    if value < 1:
      counter = counter + 1
    if value >= 0:
      if counter > 0 and counter > ser[1]:
        ser[0] = index
        ser[1] = counter
      counter = 0



  #if no dic notch found:
  try:
    if ser[0] == 0 or ser[1] < len(reg)*.05:
      #estimate diastolic pressure as flattest point -> highest diff
      #range for diastolic pressure:
      diaReg = diff.tail(n=round(len(diff)*.9))
      diaP = diaReg.idxmax()
      dicN = round(mean([diaReg.idxmax(),diaReg.idxmin()]))
      #print("NO DIC NOTCH FOUND, estimating...")
    else:
      dicN = ser[0]
      if round(2*sysMaxi) > dicN+1:
        diaP = wave[dicN+1:round(2*sysMaxi)].idxmax()
      else:
        diaP = dicN+1
  except:
    return


  NP[0] = dicN
  NP[1] = diaP

  #saving dic notch and diastolic pressure for functions
  dicNotch = NP[0]
  diaP = NP[1]

  #beginning and end of wave
  beg = 0
  end = wave.last_valid_index()
  sysMaxi = wave.idxmax()


  #add calculated metrics to lists
  pp_pres = np.sum(wave)
  avg_sys_rise = wave[beg:sysMaxi].mean()
  sys_rise_area = sum(wave[beg:sysMaxi])
  t_sys_rise = sysMaxi
  avg_dec = wave[sysMaxi:end].mean()
  t_dec = end - sysMaxi
  dec_area = np.sum(wave[sysMaxi:end])
  avg_sys = wave[beg:dicNotch].mean()
  slope_sys = (wave[dicNotch] - wave[beg]) / dicNotch
  sys_area = sum(wave[beg:dicNotch])
  t_sys = dicNotch
  avg_sys_dec = wave[sysMaxi:dicNotch].mean()
  dn_sys = wave[sysMaxi] - wave[dicNotch]
  sys_dec_area = np.sum(wave[sysMaxi:dicNotch])
  t_sys_dec = dicNotch - sysMaxi
  avg_sys_dec_nodia = wave[sysMaxi:dicNotch].mean() - wave[diaP]
  avg_sys_nodia = wave[beg:dicNotch].mean() - wave[diaP]
  avg_sys_rise_nodia = wave[beg:sysMaxi].mean() - wave[diaP]
  avg_dec_nodia = wave[sysMaxi:end].mean() - wave[diaP]
  slope_dia = (wave[end] - wave[dicNotch]) / (end - dicNotch)
  t_dia = end - dicNotch
  avg_dia = wave[dicNotch:end].mean()
  dn_dia = wave[diaP] - wave[dicNotch]

  return [pp_pres,avg_sys_rise,sys_rise_area,t_sys_rise,avg_dec,t_dec,dec_area,avg_sys,slope_sys,sys_area,t_sys,avg_sys_dec,dn_sys,sys_dec_area,t_sys_dec,avg_sys_dec_nodia,avg_sys_nodia,avg_sys_rise_nodia,avg_dec_nodia,slope_dia,t_dia,avg_dia,dn_dia,avg_sys_nodia]

def analyzeWave(waveformData, segmentIndices):

  metrics = pd.DataFrame()

  for j in range(len(waveformData)): # loops over all of patients
    for i in range(len(segmentIndices[j])-1): # loops over all segments of patient wave
      # print("{a} {b}".format(a=j, b=i))
      lowerB = segmentIndices[j][i]
      upperB = segmentIndices[j][i+1]
      wave = waveformData[j][lowerB:upperB]
      stats = calcStats(wave)

      try:
        stats = [j] + stats
        stats = pd.DataFrame(stats).transpose()
        metrics = metrics.append(stats, ignore_index = True)
      except:
        pass
  
  return metrics
