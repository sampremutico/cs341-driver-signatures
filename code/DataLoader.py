#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from torch.autograd import Variable
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from datetime import datetime
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

DATA_DIR = "../data/cs341-driver-data/nervtech/v1/drives-with-collisions/"
DATA = "user_1636_scenario_0_repeat_0_opti.csv"
DROP_LIST = set(["RAIN", "SNOW", "AREA_ID", "SCENARIO_ID", "MODALITY_ID", "PHONE_CALL", "SUBTASK_NUMBER", "SUBTASK_ACTION", "NEXT_VEHICLE_ID"])

READING_HZ = 30

class DataLoader():
	# load indicates whether to load csv into raw dataframe but this is slow
	# so can load from pickled df if already done (this just has the unnecessary rows dropped but no labels added)
	def __init__(self, ride_file=None, load=True):
		self.csv = DATA_DIR + ride_file
		self.user = DATA[5:9]
		if load == False:
			self.df = pd.read_csv(self.csv)
			print("Dropping {} columns".format(len(self.df.columns)))
			self.df = self.df.drop(DROP_LIST, axis=1)
			self.num_rows = self.df.shape[0]
			self.df.to_pickle(path=self.user+"_raw_ride.pkl")
		else:
			self.df = pd.read_pickle(self.user+"_raw_ride.pkl")
			self.num_rows = self.df.shape[0]

	# given a row and the row that indicates that previous recorded crash
	# determines whether new crash occurs in this row
	def __detection_fn(self, row, prev_crash, crash_cnt = 0):

		if prev_crash is None:
			return 1, row

		curr_x, curr_y, curr_col_id_1, curr_col_id_2 = row["POSITION_X"], row["POSITION_Y"], row["COLLISION_ID_1"], row["COLLISION_ID_2"]
		prev_crash_x, prev_crash_y, prev_crash_id_1, prev_crash_id_2 = prev_crash["POSITION_X"], prev_crash["POSITION_Y"], prev_crash["COLLISION_ID_1"], prev_crash["COLLISION_ID_2"]

		distance_thresh = 10
		curr_coords = np.array([curr_x, curr_y])
		prev_crash_coords = np.array([prev_crash_x, prev_crash_y])
		curr_dist = np.linalg.norm(prev_crash_coords - curr_coords)
		prev_time = datetime.utcfromtimestamp(prev_crash["TIMESTAMP"]/1000)
		curr_time = datetime.utcfromtimestamp(row["TIMESTAMP"]/1000)
		time_diff = (curr_time - prev_time).total_seconds()

		if curr_dist > distance_thresh and time_diff > 15 and (curr_col_id_1 != prev_crash_id_1 or curr_col_id_2 != prev_crash_id_2):
				prev_time = datetime.utcfromtimestamp(prev_crash["TIMESTAMP"]/1000)
				curr_time = datetime.utcfromtimestamp(row["TIMESTAMP"]/1000)
				
				print("found crash {}".format(crash_cnt))
				print("Prev time: --> " +prev_time.strftime('%Y-%m-%d %H:%M:%S.%f'))
				print("Curr time: --> " +curr_time.strftime('%Y-%m-%d %H:%M:%S.%f'))
				print("seconds: ", time_diff, "distance: ", curr_dist)
				print("Prev row: ", prev_crash_x, prev_crash_y, prev_crash_id_1, prev_crash_id_2)
				print("Curr row: ", curr_x, curr_y, curr_col_id_1, curr_col_id_2)
				
				return 1, row
		return 0, prev_crash 

	# adds column with 1 indicating new crash occuring at reading, 0 if no crash
	# --> uses __detection_fn as a heuristic for detecting distinct crash
	def __generate_labels(self):
		crash_labels = np.zeros(self.num_rows)
		crash_cnt = 0
		prev_crash = None
		for index, row in self.df.iterrows():
			if index == 0:
				continue
			if index % 10000 == 0:
				print(index)
			crash_labels[index], prev_crash = self.__detection_fn(row, prev_crash, crash_cnt)
			crash_cnt += crash_labels[index]
		return crash_labels

	# generates a column with 1 indicated if a crash occurs WITHIN pred_window_secs of the current reading
	# in the future (these will then be used as the labels in our sequences)
	def __generate_sequences(self, pred_window_secs):
		row_to_add = np.zeros(self.num_rows)
		for idx, row in self.df.iterrows():
			if row['crash_label'] == 1:
				for sub in range(-READING_HZ*pred_window_secs,0,1):
					if idx + sub > 0:
						row_to_add[idx+sub] = 1
					
		print(sum(row_to_add))
		return row_to_add

	# adds appropriate crash columns labeling distinct crashes and whether crash occurs
	# within pred_window_secs of row.
	# TODO --> actually generate size prev_window_secs sequences to feed into sequential model
	def segment_crashes(self, prev_window_secs=5, pred_window_secs=2, down_sampling=None, load=True):
		# if detection of new crashes has already been done then load this pickled dataframe
		# otherwise detect new crashes and save df
		if load:
			self.df = pd.read_pickle(self.user+"_segmented_crashes.pkl")
		else:
			print("Generating crash sequences of {} seconds and crashes within {} seconds".format(prev_window_secs, pred_window_secs))
			crash_label = np.zeros(self.num_rows)
			crash_cnt = 0
			prev_crash = None
			crash_labels = self.__generate_labels()
			self.df['crash_label'] = crash_labels
			self.df.to_pickle(self.user+"_segmented_crashes.pkl")

		print(datetime.utcfromtimestamp(self.df["TIMESTAMP"][0]/1000))
		print(datetime.utcfromtimestamp(self.df["TIMESTAMP"][self.num_rows-1]/1000))
		print(np.sum(self.df['crash_label']))
		# now that we have detected crashes, create sequence labels
		sequences = self.__generate_sequences(pred_window_secs)
		self.df['crash_within_pred_window'] = sequences
		# TODO: still need to actually create sequences from sequence labels now in dataframe
		return sequences

dl = DataLoader(DATA, load=False)
dl.segment_crashes(load=False)

