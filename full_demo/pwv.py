import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from statistics import *
import heartpy as hp
import os
from typing import List
from sklearn.model_selection import train_test_split
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer


def filterWave(data: np.ndarray, sample_rate: float = 240.0, bpmin: float = 0, bpmax: float = 550, lowpass: bool = True, highpass: bool = True, returnPlot: List[bool] = [False, False], patPlotShow: int = 0):
  """Filters a patient's entire waveform.

  Args:
    data: Data for a single patient's waveform
    sample_rate: Sample rate of data collected.
    bpmin: Min blood pressure.
    bpmax: Max blood pressure.
    lowpass: If True, uses a lowpass Butterworth filter on data.
    highpass: If True, uses a highpass Butterworth filter on data.
    returnPlot: List of Booleans. If True, returns a matplotlib plot of corresponding data. Bool 1 = unfiltered waveform. Bool 2 = filtered waveform.
    patPlotShow: Index of patient whose data will be shown in plots
  Returns:
    wd: Data from process method in heartpy. See heartpy docs.
    m: Metrics from process method in heartpy. See heartpy docs.
    plots: list of plots generated.
  """
  # Change values ouside possible range to min and max pulse value
  data = np.array([bpmin if i <= bpmin else (bpmax if i > bpmax else i) for i in data])
  plots = []

  if(returnPlot[0]):
    wd, m = hp.process(hrdata=data, sample_rate=sample_rate, bpmmin=bpmin, bpmmax=bpmax)
    ax = plt.figure(figsize=(16,7))
    plt.plot(wd['hr'])
    plt.xlabel("Time")
    plt.ylabel("PWV")
    plt.title("Pulse Wave Signal: Patient " + str(patPlotShow))
    plots.append(ax)
    plt.close(ax)

  data = hp.filter_signal(data, cutoff=15, sample_rate=sample_rate, order=4, filtertype='lowpass') if lowpass else data
  data = hp.filter_signal(data, cutoff=.01, sample_rate=sample_rate, order=4, filtertype='highpass') if highpass else data
  wd, m = hp.process(hrdata=data, sample_rate=sample_rate, bpmmin=bpmin, bpmmax=bpmax)

  if(returnPlot[1]):
    ax1 = plt.figure(figsize=(16,7))
    plt.plot(wd['hr'])
    plt.xlabel("Time")
    plt.ylabel("PWV")
    plt.title("Pulse Wave Signal: Patient " + str(patPlotShow))
    plots.append(ax1)
    plt.close(ax1)

  return wd, m, plots

def segmentWave(peakList: List[int]):
  """Segments waveform.

  Args:
    peakList: Peaklist returned from process method in heartpy. See heartpy docs.
  Returns:
    rngs: List of segment indices for waveform.
  """
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

def preprocess(dataDir: str, sample_rate: float = 240.0, bpmin: float = 0, bpmax: float = 550, lowpass: bool = True, highpass: bool = True, returnPlot: List[bool] = [False, False, False], patPlotShow: int = 0):
  """Preprocessing, through filtering and segmentation, of a patient's pulsewave.

  Args:
    dataDir: Directory in which data files are stored. Reqs: full file path, includes only .txt, .csv, or .mat files with the column_name 'AO' for data.
    sample_rate: Sample rate of data collected.
    bpmin: Min blood pressure.
    bpmax: Max blood pressure.
    lowpass: If True, uses a lowpass Butterworth filter on data.
    highpass: If True, uses a highpass Butterworth filter on data.
    returnPlot: List of Booleans. If True, returns a matplotlib plot of corresponding data. Bool 1 = unfiltered waveform. Bool 2 = filtered waveform. Bool 3 = segmented waveform.
    patPlotShow: Index of patient whose data will be shown in plots

  Returns:
    waveformData: Pulse wave data for all patients. Each list is for one patient.
    segmentIndices: Segementation indices for all patients. Each list is for one patient.
    plots: list of plots generated.
  """
  path = dataDir
  list_of_files = []

  for root, _, files in os.walk(path):
    for file in files:
      list_of_files.append(os.path.join(root,file))
  list_of_files.sort()
  
    
  waveformData = [] # waveform data
  segmentIndices = [] # segmentation indices
  plots = []

  for i in range(len(list_of_files)):

    data = hp.get_data(list_of_files[i], delim = ' ', column_name = 'AO')
    wd, _, ax = filterWave(data, sample_rate, bpmin, bpmax, lowpass, highpass, returnPlot[0:2], patPlotShow) if i==patPlotShow else filterWave(data, sample_rate, bpmin, bpmax, lowpass, highpass)
    plots = plots + ax
    waveformData.append(wd['hr'])
    segmentIndices.append(segmentWave(wd['peaklist']))
  
  if(returnPlot[2]):
    displayLen = min(len(waveformData[patPlotShow]), 4000)
    ax = plt.figure(figsize=(16,7))
    plt.plot(waveformData[patPlotShow][0:displayLen])
    plt.xlabel("Time")
    plt.ylabel("PWV")
    plt.title("Pulse Wave Signal: Patient " + str(patPlotShow))
    plots.append(ax)
    plt.close(ax)

    ax1 = plt.figure(figsize=(16,7))
    plt.plot(waveformData[patPlotShow][0:displayLen])
    plt.xlabel("Time")
    plt.ylabel("PWV")
    plt.title("Pulse Wave Signal: Patient " + str(patPlotShow))
    for xc in segmentIndices[patPlotShow]:
      if(xc>displayLen):
        break
      plt.axvline(x=xc, color='purple', linestyle="--")
    plots.append(ax1)
    plt.close(ax1)

  return waveformData, segmentIndices, plots

def plotSegment(waveformData: List[List[float]], segmentIndices: List[List[int]], patNum: List[int] = [0], segNum: List[int] = [0]):
  """Plots waveform segments.

  Args:
    waveformData: Pulse wave data for all patients. Each list is for one patient.
    segmentIndices: Segementation indices for all patients. Each list is for one patient.
    patNum: List of patients to plot.
    segNum: List of segments to plot.
  Returns:
    patPlots_df: dataframe of patient and segement plots
  """
  patPlots = {}
  for pat in patNum:
    patPlots["pat"+str(pat)] = {}
    for seg in segNum:
      lowerB = segmentIndices[pat][seg]
      upperB = segmentIndices[pat][seg+1]
      wave = waveformData[pat][lowerB:upperB]

      ax = plt.figure(figsize=(16,7))
      plt.plot(wave)
      plt.xlabel("Time")
      plt.ylabel("PWV")
      plt.title("PWV: Patient " + str(pat) + ", Segment " + str(seg))
      patPlots["pat"+str(pat)]["seg" + str(seg)] = ax
      plt.close(ax)

  patPlots_df = pd.DataFrame(patPlots).transpose()
  return patPlots_df


def calcStats(wave: List[float], verbose: bool = False):
  """Calculates waveform stats of given wave.

  Args:
    wave: a single waveform.
  Returns:
    metList: list of all calculated metrics
      can return None if a metric fails to be calculated
    points: dict of indexes of 5 main points
  """
  #calculate metrics for each indexed wave saved in individual metric arrays
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
  #avg_dia_nodia = wave[dicNotch:end].mean() - wave[diaP] 

  points = {
      "Start": beg,
      "Max": sysMaxi,
      "Dic Notch": dicNotch,
      "Dia Pressure": diaP,
      "End": end
  }
    
  metList=[pp_pres,avg_sys_rise,sys_rise_area,t_sys_rise,avg_dec,t_dec,dec_area,avg_sys,slope_sys,sys_area,t_sys,avg_sys_dec,dn_sys,sys_dec_area,t_sys_dec,avg_sys_dec_nodia,avg_sys_nodia,avg_sys_rise_nodia,avg_dec_nodia,slope_dia,t_dia,avg_dia,dn_dia,avg_sys_nodia]
  return metList, points


def interPlotSegment(waveformData: List[List[float]], segmentIndices: List[List[int]], patNum: List[int] = [0], segNum: List[int] = [0]):
  """Plots INTERACTIVE waveform segments.

  Args:
    waveformData: Pulse wave data for all patients. Each list is for one patient.
    segmentIndices: Segementation indices for all patients. Each list is for one patient.
    patNum: List of patients to plot.
    segNum: List of segments to plot.
  Returns:
    patPlots_df: dataframe of patient and segement plots
    patPoints_df: dict of 5 indexes
  """
  patPlots = {}
  patPoints = {}
  for pat in patNum:
    patPlots["pat"+str(pat)] = {}
    patPoints["pat"+str(pat)] = {}
    for seg in segNum:
      lowerB = segmentIndices[pat][seg]
      upperB = segmentIndices[pat][seg+1]
      wave = waveformData[pat][lowerB:upperB]
    
      metrics_df, points = calcStats(wave)
      
      plt.rcParams["figure.figsize"]=12,10
      fig = plt.figure()
      ax = fig.add_subplot(111)
      
      wav, = ax.plot(wave, '-', label='wave')
      start, = ax.plot(points["Start"], wave[points["Start"]], 'o', label='Beg')
      max, = ax.plot(points["Max"], wave[points["Max"]], 'o', label='Max')
      dic, = ax.plot(points["Dic Notch"], wave[points["Dic Notch"]], 'o', label='Dic Notch')
      dia, = ax.plot(points["Dia Pressure"], wave[points["Dia Pressure"]], 'o', label='Dia Pressure')
      end, = ax.plot(points["End"], wave[points["End"]], 'o', label='End')
      
      #lte = [*range(len(wave))] 
      #pp_pres, = ax.fill_between(lte, wave, label='pp_pres')
      
      avg_sys_rise, = ax.plot(np.array([0,points["Max"]]), np.array([metrics_df[1],metrics_df[1]]), label='avg_sys_rise')
      #sys_rise_area,
      t_sys_rise, = ax.plot(np.array([0,points["Max"]]), np.array([wave[points["Start"]],wave[points["Start"]]]), '--',label='t_sys_rise')
      avg_dec, = ax.plot(np.array([points["Max"],points["End"]]), np.array([metrics_df[4],metrics_df[4]]), label='avg_dec')
      t_dec, = ax.plot(np.array([points["Max"],points["End"]]), np.array([wave[points["End"]],wave[points["End"]]]), '--',label='t_sys_dec')
      #dec_area,
      avg_sys, = ax.plot(np.array([0,points["Dic Notch"]]), np.array([metrics_df[7],metrics_df[7]]), label='avg_sys')
      slope_sys, = ax.plot(np.array([0,points["Dic Notch"]]), np.array([wave[points["Start"]],wave[points["Dic Notch"]]]), '-.',label='slope_sys')
      #sys_area,
      t_sys, = ax.plot(np.array([0,points["Dic Notch"]]), np.array([wave[points["Dic Notch"]],wave[points["Dic Notch"]]]), '--',label='t_sys')
      avg_sys_dec, = ax.plot(np.array([points["Max"],points["Dic Notch"]]), np.array([metrics_df[11],metrics_df[11]]), label='avg_sys_dec')
      dn_sys, = ax.plot(np.array([points["Max"],points["Max"]]), np.array([wave[points["Max"]],wave[points["Dic Notch"]]]), '--', label='dn_sys')
      #sys_dec_area,
      t_sys_dec, = ax.plot(np.array([points["Max"],points["Dic Notch"]]), np.array([wave[points["Dic Notch"]],wave[points["Dic Notch"]]]), '--',label='t_sys_dec')
      avg_sys_dec_nodia, = ax.plot(np.array([points["Max"],points["Dic Notch"]]), np.array([metrics_df[15],metrics_df[15]]), label='avg_sys_dec_nodia')
      avg_sys_nodia, = ax.plot(np.array([0,points["Dic Notch"]]), np.array([metrics_df[16],metrics_df[16]]), label='avg_sys_nodia')
      avg_sys_rise_nodia, = ax.plot(np.array([0,points["Max"]]), np.array([metrics_df[17],metrics_df[17]]), label='avg_sys_rise_nodia')
      avg_dec_nodia, = ax.plot(np.array([points["Max"], points["End"]]), np.array([metrics_df[18],metrics_df[18]]), label='avg_dec_nodia')
      slope_dia, = ax.plot(np.array([points["Dic Notch"],points["End"]]), np.array([wave[points["Dic Notch"]],wave[points["End"]]]), '-.',label='slope_dia')
      t_dia, = ax.plot(np.array([points["Dic Notch"],points["End"]]), np.array([wave[points["End"]],wave[points["End"]]]), '--', label='t_dia')
      avg_dia, = ax.plot(np.array([points["Dic Notch"],points["End"]]), np.array([metrics_df[21],metrics_df[21]]), label='avg_dia')
      dn_dia, = ax.plot(np.array([points["Dia Pressure"],points["Dia Pressure"]]), np.array([wave[points["Dia Pressure"]],wave[points["Dic Notch"]]]), '--', label='dn_dia')
      avg_dia_nodia, = ax.plot(np.array([points["Dic Notch"],points["End"]]), np.array([metrics_df[23],metrics_df[23]]), label='avg_dia_nodia')
      
      legend = ax.legend(bbox_to_anchor=(1, .4))
      plt.xlabel("Time")
      plt.ylabel("PWV")
      plt.title("PWV: Patient " + str(pat) + ", Segment " + str(seg))
    
    
      wave_leg,beg_leg,max_leg,dic_notch_leg,dia_pres_leg,end_leg,avg_sys_rise_leg,t_sys_rise_leg,avg_dec_leg,t_dec_leg,avg_sys_leg,slope_sys_leg,t_sys_leg,avg_sys_dec_leg,dn_sys_leg,t_sys_dec_leg,avg_sys_dec_nodia_leg,avg_sys_nodia_leg,avg_sys_rise_nodia_leg,avg_dec_nodia_leg,slope_dia_leg,t_dia_leg,avg_dia_leg,dn_dia_leg,avg_dia_nodia_leg = legend.get_lines()
      wave_leg.set_picker(True)
      wave_leg.set_pickradius(5)
        

      pickables = {}
      pickables[wave_leg] = wave
      

      def on_pick(event):
          leg = event.artist
          visible = leg.get_visible()
          visible = not visible
          pickables[leg].set_visible(visible)
          leg.set_visible(visible)
          fig.canvas.draw()

      plt.connect('pick_event', on_pick)    
    
    
    
    
    
    
    
    
    
    
    
      
      patPlots["pat"+str(pat)]["seg" + str(seg)] = fig
      patPoints["pat"+str(pat)]["seg" + str(seg)] = points

    
      plt.close(fig)
        



  patPlots_df = pd.DataFrame(patPlots).transpose()
  patPoints_df = pd.DataFrame(patPoints).transpose()

  return patPlots_df, patPoints_df, metrics_df




def analyzeWave(waveformData: List[List[float]], segmentIndices: List[List[int]]):
  """Calls calcStats for every segment of every patient.

  Args:
    waveformData: Pulse wave data for all patients. Each list is for one patient.
    segmentIndices: Segementation indices for all patients. Each list is for one patient.
  Returns:
    metrics: dataframe of patient metrics
  """
  metrics = pd.DataFrame()

  for j in range(len(waveformData)): # loops over all of patients
    for i in range(len(segmentIndices[j])-1): # loops over all segments of patient wave
      # print("{a} {b}".format(a=j, b=i))
      lowerB = segmentIndices[j][i]
      upperB = segmentIndices[j][i+1]
      wave = waveformData[j][lowerB:upperB]
      
      #stats, _ = calcStats(wave)

    
      try:
        stats, _ = calcStats(wave)
        stats = [j] + stats
        stats = pd.DataFrame(stats).transpose()
        metrics = metrics.append(stats, ignore_index = True)
      except:
        pass
  
  return metrics

def MLSplit(outcomeData: pd.DataFrame()):
  """Splits data into testing and training sets.

  Args:
    outcomeData: Waves with metrics and outcomes.
  Returns:
    x_train, x_test, y_train, y_test: training and testing data
  """
  x = outcomeData.iloc[:,1:25]
  y = outcomeData.iloc[:,25]

  x_train, x_test, y_train, y_test = train_test_split(x, y)
  
  imp = SimpleImputer(missing_values=np.nan, strategy='mean')
  imp = imp.fit(x_train)
  x_train_imp = imp.transform(x_train)

  return x_train_imp, x_test, y_train, y_test

def logistic(x_train, x_test, y_train, y_test):
  model = LogisticRegression(max_iter=4000)
  model.fit(x_train, y_train)

  y_pred = pd.Series(model.predict(x_test))
  y_test = y_test.reset_index(drop=True)
  z = pd.concat([y_test, y_pred], axis=1)
  z.columns = ['True', 'Prediction']
  accuracy = metrics.accuracy_score(y_test, y_pred)
  precision = metrics.precision_score(y_test, y_pred)
  recall = metrics.recall_score(y_test, y_pred)
    
  return z, accuracy, precision, recall

def decisionTree(x_train, x_test, y_train, y_test):
  classifier = DecisionTreeClassifier()
  classifier.fit(x_train, y_train)

  y_pred = pd.Series(classifier.predict(x_test))
  y_test = y_test.reset_index(drop=True)
  z = pd.concat([y_test, y_pred], axis=1)
  z.columns = ['True', 'Prediction']
  accuracy = metrics.accuracy_score(y_test, y_pred)
  precision = metrics.precision_score(y_test, y_pred)
  recall = metrics.recall_score(y_test, y_pred)
    
  return z, accuracy, precision, recall

def svm(x_train, x_test, y_train, y_test):
  classifier = svm.SVC()
  classifier.fit(x_train, y_train)

  y_pred = pd.Series(classifier.predict(x_test))
  y_test = y_test.reset_index(drop=True)
  z = pd.concat([y_test, y_pred], axis=1)
  z.columns = ['True', 'Prediction']
  accuracy = metrics.accuracy_score(y_test, y_pred)
  precision = metrics.precision_score(y_test, y_pred)
  recall = metrics.recall_score(y_test, y_pred)
    
  return z, accuracy, precision, recall

def KNeighbors(x_train, x_test, y_train, y_test):
  classifier = KNeighborsClassifier(n_neighbors=3)
  classifier.fit(x_train, y_train)

  y_pred = pd.Series(classifier.predict(x_test))
  y_test = y_test.reset_index(drop=True)
  z = pd.concat([y_test, y_pred], axis=1)
  z.columns = ['True', 'Prediction']
  accuracy = metrics.accuracy_score(y_test, y_pred)
  precision = metrics.precision_score(y_test, y_pred)
  recall = metrics.recall_score(y_test, y_pred)
    
  return z, accuracy, precision, recall

def gaussianNB(x_train, x_test, y_train, y_test):
  classifier = GaussianNB()
  classifier.fit(x_train, y_train)

  y_pred = pd.Series(classifier.predict(x_test))
  y_test = y_test.reset_index(drop=True)
  z = pd.concat([y_test, y_pred], axis=1)
  z.columns = ['True', 'Prediction']
  accuracy = metrics.accuracy_score(y_test, y_pred)
  precision = metrics.precision_score(y_test, y_pred)
  recall = metrics.recall_score(y_test, y_pred)
    
  return z, accuracy, precision, recall
