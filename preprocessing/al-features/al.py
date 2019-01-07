#!/usr/bin/python
#
# Compute AL features (code from Dr. Cook)
#
# Usage: python2 compute_al_features.py al.config file.txt output.hdf5
#
import gzip
import numpy as np
import os
import sys
import h5py
import time
import datetime
from datetime import datetime
from datetime import timedelta
from sklearn import model_selection, tree
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# list of global variables and initial values
activitynames = []
currentSecondsOfDay = 0
currentTimestamp = 0
dayOfWeek = 0
dominant = 0
gFeatureNames = []
sensornames = []
sensortimes = []
clf = []
kmeans = []
data = []
dstype = []
labels = []
numwin = 0
prevwin1 = 0
prevwin2 = 0
wincnt = 0
n_clusters = 2
DataFileName = "data"
OutputFileName = "data.hdf5"
IgnoreOther = 0
ClusterOther = 0
NoOverlap = 1 # Do not allow windows to overlap
Mode = "TRAIN"   # TRAIN, TEST, CV, ANNOTATE, WRITE
ModelName = "model"
NumActivities = 0
NumClusters = 10
NumData = 0
NumSensors = 0
NumSetFeatures = 14
SecondsInADay = 86400
MaxWindow = 30
weightinc = 0.01
windata = np.zeros((MaxWindow, 3), dtype=np.int)

# list of classifier names and parameters
c1 = RandomForestClassifier(n_estimators=60, max_features=8, bootstrap=True, criterion="entropy", min_samples_split=20, max_depth=None, n_jobs=4, class_weight='balanced')
classifiers = [
   (c1, "RF")
]

def ReadConfig(name):
   global NumSensors, NumActivities
   global sensornames, sensortimes, activitynames, dstype
   global IgnoreOther, ClusterOther, DataFileName, Mode, ModelName

   configfile = open(name, "r")
   for line in configfile:
      words = line.split()  # Split line into words on delimiter " "
      if len(words) == 0:
         print "Config file is empty"
         exit()
      else:
         if line[0] == "%":          # Ignore comment line
            words = []
         elif words[0] == "data":   # Data file name
            DataFileName = words[1]
         elif words[0] == "mode": # AL mode
            Mode = words[1]
         elif words[0] == "ignoreother":
            IgnoreOther = 1
         elif words[0] == "clusterother":
            IgnoreOther = 0
            ClusterOther = 1
         elif words[0] == "model":
            ModelName = words[1]
         elif words[0] == "sensors":
            for i in range(1,len(words)):
               sensornames.append(words[i])
               sensortimes.append(0)
               dstype.append('n')
            NumSensors = len(sensornames)
         elif words[0] == "activities":
            for i in range(1,len(words)):
               activitynames.append(words[i])
            NumActivities = len(activitynames)
   configfile.close()

def SetFeatureNames():
   gFeatureNames.append("lastSensorEventHours")
   gFeatureNames.append("lastSensorEventSeconds")
   gFeatureNames.append("lastSensorDayOfWeek")
   gFeatureNames.append("windowDuration")
   gFeatureNames.append("timeSinceLastSensorEvent")
   gFeatureNames.append("prevDominantSensor1")
   gFeatureNames.append("prevDominantSensor2")
   gFeatureNames.append("lastSensorID")
   gFeatureNames.append("lastSensorLocation")
   gFeatureNames.append("lastMotionLocation")
   gFeatureNames.append("complexity")
   gFeatureNames.append("activityChange")
   for i in range(NumSensors):
      gFeatureNames.append("sensorCount-" + sensornames[i])
   for i in range(NumSensors):
      gFeatureNames.append("sensorElTime-" + sensornames[i])

# convert string date and time values to a datetime structure
def get_datetime(date, time):
   dtstr = date + ' ' + time
   try:
      dt = datetime.strptime(dtstr, "%Y-%m-%d %H:%M:%S.%f")
   except:
      dt = datetime.strptime(dtstr, "%Y-%m-%d %H:%M:%S")
   return dt


# Compute the number of seconds past midnight for the current datetime
def ComputeSeconds(dt):
   seconds = dt - dt.replace(hour=0, minute=0, second=0)
   return int(seconds.total_seconds())


# Return the index of a specific sensor name in the list of sensors
def FindSensor(sensorname):
   try:
      i = sensornames.index(sensorname)
      return i
   except:
      print "Could not find sensor ", sensorname
      return -1


# Return the index of a specific activity name in the list of activities
# If activity is not found in the list, add the new name
def FindActivity(aname):
   global activitynames, NumActivities

   try:
      i = activitynames.index(aname)
      return i
   except:
      print "Warning: could not find activity", aname, "-- will add to list"
      if Mode == "TEST":
         print "Could not find activity ", aname
         return -1
      else:
         activitynames.append(aname)
         NumActivities += 1
         return NumActivities - 1


def ReadData():
   global currentSecondsOfDay, dayOfWeek, currentTimestamp, dstype
   global NumData, data, wincnt

   #sameActivityCount = 0
   #sameActivityLabel = "Other_Activity"
   firstEventFlag = 1
   SetFeatureNames()
   currentTimestamp = get_datetime("2001-01-01", "00:00:00.00000")

   datafile = open(DataFileName, "r")
   count = 0
   for line in datafile:
      words = line.split()  # Split line into words on delimiter " "
      if len(words) == 5:
          date = words[0]
          stime = words[1]
          sensorid = words[2]
          newsensorid = words[2]
          sensorstatus = words[3]
          alabel = words[4]
      else:
          date = words[0]
          stime = words[1]
          sensorid = words[2]
          newsensorid = words[3]
          sensorstatus = words[4]
          alabel = words[5]

      dt = get_datetime(date, stime)
      currentSecondsOfDay = ComputeSeconds(dt)
      dayOfWeek = dt.weekday()
      previousTimestamp = currentTimestamp
      currentTimestamp = dt

      snum1 = FindSensor(sensorid)

      timediff = currentTimestamp - previousTimestamp

      # reset the sensor times and the window
      if firstEventFlag == 1 or timediff.days < 0 or timediff.days > 1:
         for i in range(NumSensors):
            sensortimes[i] = currentTimestamp - timedelta(days=1)
         firstEventFlag = 0
         wincnt = 0

      if sensorstatus == "ON" and dstype[snum1] == 'n':
         dstype[snum1] = 'm'

      sensortimes[snum1] = currentTimestamp  # last time sensor fired

      tempdata = np.zeros(NumFeatures)

      end = 0
      if alabel != "Other_Activity" or IgnoreOther == 0:
         #(sameActivityCount > (MaxWindow / 2)):
         end = ComputeFeature(dt, sensorid, newsensorid, tempdata)

      if end == 1:  # End of window reached, add feature vector
         if alabel == "Other_Activity" and ClusterOther == 1:
            labels.append(-1)
         else:
            labels.append(FindActivity(alabel))
         data.append(tempdata)
         count += 1

   datafile.close()
   if Mode != "TEST":
      if ClusterOther == 1:
         ClusterOtherClass()
   NumData = len(data)

   # Save the features and labels for use later and exit
   # np.save(OutputFileName, {
   #     "features": np.array(data),
   #     "labels": np.array(labels),
   # })
   d = h5py.File(OutputFileName, "w")
   d.create_dataset("features", data=np.array(data), compression="gzip")
   d.create_dataset("labels", data=np.array(labels), compression="gzip")

# Cluster the Other_Activity class into subclasses
def ClusterOtherClass():
   global kmeans, NumActivities, NumClusters

   carray = []
   for i in range(len(labels)):
      if labels[i] == -1:  # Other activity
         carray.append(data[i])
   kmeans = MiniBatchKMeans(n_clusters=NumClusters).fit(carray)
   clabels = kmeans.labels_
   # The actual number of resulting clusters
   newlabels = kmeans.predict(carray)
   #numclusters = len(set(labels)) - (1 if -1 in labels else 0)
   numclusters = len(set(newlabels))
   NumClusters = numclusters
   for i in range(NumClusters):
      activitynames.append('cluster_' + str(i))

   # assign Other activity labels new subclasses
   index = 0
   for i in range(len(labels)):
      if labels[i] == -1:  # Other activity
         labels[i] = newlabels[index] + NumActivities
         index += 1
   with gzip.GzipFile('./model/clusters.gz', 'wb') as f:
      joblib.dump(kmeans, f)   # save clusters to file
   #joblib.dump(kmeans, './model/clusters.pkl')   # save clusters to file
   NumActivities += NumClusters


# Compute the feature vector for each window-size sequence of sensor events.
# 0: time of the last sensor event in window (hour)
# 1: time of the last sensor event in window (seconds)
# 2: day of the week for the last sensor event in window
# 3: window size in time duration
# 4: time since last sensor event
# 5: dominant sensor for previous window
# 6: dominant sensor two windows back
# 7: last sensor event in window
# 8: last sensor location in window
# 9: last motion sensor location in window
# 10: complexity of window (entropy calculated from sensor counts)
# 11: change in activity level between two halves of window
# 12: number of transitions between areas in window
# 13: number of distinct sensors in window
# 14 - NumSensors+13: counts for each sensor
# NumSensors+14 - 2*NumSensors+13: time since sensor last fired (<= SECSINDAY)

def ComputeFeature(dt, sensorid1, sensorid2, tempdata):
   global wincnt, prevwin1, prevwin2, numwin, dominant

   lastlocation = -1
   lastmotionlocation = -1
   complexity = 0
   maxcount = 0
   numtransitions = 0
   numdistinctsensors = 0

   windata[wincnt][0] = FindSensor(sensorid1)
   windata[wincnt][1] = currentSecondsOfDay
   windata[wincnt][2] = dayOfWeek

   if wincnt < (MaxWindow-1):   # not reached end of window
      wincnt += 1
      return 0
   else:                        # reached end of window
      wsize = MaxWindow
      scount = np.zeros(NumSensors, dtype=np.int)

      # Determine the dominant sensor for this window
      # count the number of transitions between areas in this window
      for i in range(MaxWindow-1, MaxWindow-(wsize+1), -1):
         scount[windata[i][0]] += 1
         id = windata[i][0]
         if lastlocation == -1:
            lastlocation = id
         if (lastmotionlocation == -1) and (dstype[id] == 'm'):
            lastmotionlocation = id
         if (i < MaxWindow-1): # check for transition
            id2 = windata[i+1][0]
            if (id != id2):
               if (dstype[id] == 'm') and (dstype[id2] == 'm'):
                  numtransitions += 1

      for i in range(NumSensors):
         if scount[i] > 1:
            ent = float(scount[i]) / float(wsize)
            ent *= np.log2(ent)
            complexity -= float(ent)
            numdistinctsensors += 1

      if np.mod(numwin, MaxWindow) == 0:
         prevwin2 = prevwin1
         prevwin1 = dominant
         dominant = 0
         for i in range(NumSensors):
            if scount[i] > maxcount:
               maxcount = scount[i]
               dominant = i

      # Attribute 0..2: time of last sensor event in window
      tempdata[0] = windata[MaxWindow-1][1] / 3600  # hour of day
      tempdata[1] = windata[MaxWindow-1][1]         # seconds of day
      tempdata[2] = windata[MaxWindow-1][2]         # day of week

      # Attribute 3: time duration of window in seconds
      time1 = windata[MaxWindow-1][1]      # most recent sensor event
      time2 = windata[MaxWindow-wsize][1]  # first sensor event in window
      if time1 < time2:
         duration = time1 + (SecondsInADay - time2)
      else:
         duration = time1 - time2
      tempdata[3] = duration  # window duration

      timehalf = windata[MaxWindow-(wsize/2)][1]   # halfway point
      if time1 < time2:
         duration = time1 + (SecondsInADay - time2)
      else:
         duration = time1 - time2
      if timehalf < time2:
         halfduration = timehalf + (SecondsInADay - time2)
      else:
         halfduration = timehalf - time2
      if duration == 0.0:
         activitychange = 0.0
      else:
         activitychange = float(halfduration) / float(duration)

      # Attribute 4: time since last sensor event
      time2 = windata[MaxWindow-2][1]
      if time1 < time2:
         duration = time1 + (SecondsInADay - time2)
      else:
         duration = time1 - time2
      tempdata[4] = duration

      # Attribute 5..6: dominant sensors from previous windows
      tempdata[5] = prevwin1
      tempdata[6] = prevwin2

      # Attribute 7: last sensor id in window
      tempdata[7] = FindSensor(sensorid1)

      # Attribute 8: last location in window
      tempdata[8] = lastlocation

      # Attribute 9: last motion location in window
      tempdata[9] = lastmotionlocation

      # Attribute 10: complexity (entropy of sensor counts)
      tempdata[10] = complexity

      # Attribute 11: activity change (activity change between window halves)
      tempdata[11] = activitychange

      # Attribute 12: number of transitions between areas in window
      tempdata[12] = numtransitions

      # Attribute 13: number of distinct sensors in window
      #tempdata[13] = numdistinctsensors
      tempdata[13] = 0

      # Attributes NumSetFeatures..(NumSensors+(NumSetFeatures-1))
      weight = 1
      for i in range(MaxWindow-1, MaxWindow-(wsize+1), -1):
         tempdata[windata[i][0]+NumSetFeatures] += 1 * weight
         weight += weightinc

      # Attributes NumSensors+NumSetFeatures..(2*NumSensors+(NumSetFeatures-1))
      # time since each sensor fired
      for i in range(NumSensors):
         difftime = currentTimestamp - sensortimes[i]
         # There is a large gap in time or shift backward in time
         if difftime.seconds < 0 or (difftime.days > 0):
            tempdata[NumSetFeatures+NumSensors+i] = SecondsInADay
         tempdata[NumSetFeatures+NumSensors+i] = difftime.seconds

      for i in range(MaxWindow-1):
         windata[i][0] = windata[i+1][0]
         windata[i][1] = windata[i+1][1]
         windata[i][2] = windata[i+1][2]
      numwin += 1
      if NoOverlap == 1 and Mode != "ANNOTATE":  # windows should not overlap
         wincnt = 0
      return 1
   return 0

if __name__ == "__main__":
   if len(sys.argv) > 1:    # Read config file
      ReadConfig(sys.argv[1])
      if len(sys.argv) > 2: # Read data file name
         DataFileName = sys.argv[2]
      if len(sys.argv) > 3: # Read output file name
         OutputFileName = sys.argv[3]

      print "Config", sys.argv[1], "Data", DataFileName, "Output", OutputFileName
   else:
      ReadConfig("al.config")

   NumFeatures = NumSetFeatures + (2 * NumSensors)
   ReadData()
