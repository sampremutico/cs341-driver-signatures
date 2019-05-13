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
from visualize_crashes import plot_course
import torch
import os

DATA_DIR = "../data/cs341-driver-data/nervtech/v1/drives-with-collisions/"
PYTORCH_DATA_DIR = "../data/pytorch/"
DATA = "user_1636_scenario_0_repeat_0_opti.csv"
DROP_LIST = set(["DATE","RAIN", "SNOW", "AREA_ID", "SCENARIO_ID", "MODALITY_ID", "PHONE_CALL", "SUBTASK_NUMBER", "SUBTASK_ACTION", "NEXT_VEHICLE_ID", 'GEAR_BOX_MODE', 'WHEEL_ANGLE', 'STEERING_WHEEL_TORQ', 'ENGINE_SPEED', 'SEAT_BELT', 'STEERING_WHEEL_SPEED', 'POSITION_X_AUTONOMOUS_VEHICLE_10', 'ROLL_AUTONOMOUS_VEHICLE_4', 'POSITION_Z_AUTONOMOUS_VEHICLE_10', 'ROLL_AUTONOMOUS_VEHICLE_9', 'PITCH_AUTONOMOUS_VEHICLE_10', 'ACCELERATION_AUTONOMOUS_VEHICLE_6', 'ACCELERATION_AUTONOMOUS_VEHICLE_7', 'ACCELERATION_AUTONOMOUS_VEHICLE_4', 'ACCELERATION_AUTONOMOUS_VEHICLE_5', 'ACCELERATION_AUTONOMOUS_VEHICLE_2', 'ACCELERATION_AUTONOMOUS_VEHICLE_3', 'ACCELERATION_AUTONOMOUS_VEHICLE_1', 'POSITION_X_AUTONOMOUS_VEHICLE_1', 'POSITION_X_AUTONOMOUS_VEHICLE_2', 'POSITION_X_AUTONOMOUS_VEHICLE_3', 'SPEED_AUTONOMOUS_VEHICLE_10', 'POSITION_X_AUTONOMOUS_VEHICLE_5', 'ACCELERATION_AUTONOMOUS_VEHICLE_8', 'ACCELERATION_AUTONOMOUS_VEHICLE_9', 'HEADING_AUTONOMOUS_VEHICLE_5', 'HEADING_AUTONOMOUS_VEHICLE_3', 'POSITION_X_AUTONOMOUS_VEHICLE_4', 'ROLL_AUTONOMOUS_VEHICLE_5', 'ROLL_AUTONOMOUS_VEHICLE_10', 'ACCELERATION_AUTONOMOUS_VEHICLE_10', 'HEADING_AUTONOMOUS_VEHICLE_4', 'HEADING_AUTONOMOUS_VEHICLE_7', 'POSITION_Y_AUTONOMOUS_VEHICLE_10', 'HEADING_AUTONOMOUS_VEHICLE_1', 'POSITION_X_AUTONOMOUS_VEHICLE_6', 'POSITION_Y_AUTONOMOUS_VEHICLE_9', 'POSITION_Y_AUTONOMOUS_VEHICLE_8', 'POSITION_Y_AUTONOMOUS_VEHICLE_7', 'POSITION_Y_AUTONOMOUS_VEHICLE_6', 'POSITION_Y_AUTONOMOUS_VEHICLE_5', 'POSITION_Y_AUTONOMOUS_VEHICLE_4', 'POSITION_Y_AUTONOMOUS_VEHICLE_3', 'POSITION_Y_AUTONOMOUS_VEHICLE_2', 'POSITION_Y_AUTONOMOUS_VEHICLE_1', 'POSITION_Z_AUTONOMOUS_VEHICLE_2', 'POSITION_Z_AUTONOMOUS_VEHICLE_3', 'POSITION_Z_AUTONOMOUS_VEHICLE_1', 'POSITION_Z_AUTONOMOUS_VEHICLE_6', 'POSITION_Z_AUTONOMOUS_VEHICLE_7', 'POSITION_Z_AUTONOMOUS_VEHICLE_4', 'POSITION_Z_AUTONOMOUS_VEHICLE_5', 'HEADING_AUTONOMOUS_VEHICLE_6', 'POSITION_Z_AUTONOMOUS_VEHICLE_8', 'POSITION_Z_AUTONOMOUS_VEHICLE_9', 'ROLL_AUTONOMOUS_VEHICLE_7', 'POSITION_X_AUTONOMOUS_VEHICLE_8', 'ROLL_AUTONOMOUS_VEHICLE_6', 'HEADING_AUTONOMOUS_VEHICLE_10', 'ROLL_AUTONOMOUS_VEHICLE_1', 'HEADING_AUTONOMOUS_VEHICLE_2', 'SPEED_AUTONOMOUS_VEHICLE_6', 'SPEED_AUTONOMOUS_VEHICLE_7', 'SPEED_AUTONOMOUS_VEHICLE_4', 'SPEED_AUTONOMOUS_VEHICLE_5', 'SPEED_AUTONOMOUS_VEHICLE_2', 'SPEED_AUTONOMOUS_VEHICLE_3', 'ROLL_AUTONOMOUS_VEHICLE_3', 'SPEED_AUTONOMOUS_VEHICLE_1', 'HEADING_AUTONOMOUS_VEHICLE_9', 'ROLL_AUTONOMOUS_VEHICLE_2', 'POSITION_X_AUTONOMOUS_VEHICLE_9', 'ROLL_AUTONOMOUS_VEHICLE_8', 'SPEED_AUTONOMOUS_VEHICLE_8', 'SPEED_AUTONOMOUS_VEHICLE_9', 'PITCH_AUTONOMOUS_VEHICLE_9', 'PITCH_AUTONOMOUS_VEHICLE_8', 'HEADING_AUTONOMOUS_VEHICLE_8', 'PITCH_AUTONOMOUS_VEHICLE_3', 'PITCH_AUTONOMOUS_VEHICLE_2', 'PITCH_AUTONOMOUS_VEHICLE_1', 'POSITION_X_AUTONOMOUS_VEHICLE_7', 'PITCH_AUTONOMOUS_VEHICLE_7', 'PITCH_AUTONOMOUS_VEHICLE_6', 'PITCH_AUTONOMOUS_VEHICLE_5', 'PITCH_AUTONOMOUS_VEHICLE_4'])

READING_HZ = 30

class DriverData():
	# load indicates whether to load csv into raw dataframe but this is slow
	# so can load from pickled df if already done (this just has the unnecessary rows dropped but no labels added)
	def __init__(self, ride_file=None, load=True):
		self.csv = DATA_DIR + ride_file
		self.user = DATA[5:9]
		self.X, self.Y = None, None
		if load == False:
			self.df = pd.read_csv(self.csv)
			print("Dropping {} columns".format(len(self.df.columns)))
			self.df = self.df.drop(DROP_LIST, axis=1, errors='ignore')
			print(len(self.df.columns))
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

		distance_thresh = 100
		curr_coords = np.array([curr_x, curr_y])
		prev_crash_coords = np.array([prev_crash_x, prev_crash_y])
		curr_dist = np.linalg.norm(prev_crash_coords - curr_coords)
		prev_time = datetime.utcfromtimestamp(prev_crash["TIMESTAMP"]/1000)
		curr_time = datetime.utcfromtimestamp(row["TIMESTAMP"]/1000)
		time_diff = (curr_time - prev_time).total_seconds()

		if curr_dist > distance_thresh and time_diff > 30 and (curr_col_id_1 != prev_crash_id_1 or curr_col_id_2 != prev_crash_id_2):
				prev_time = datetime.utcfromtimestamp(prev_crash["TIMESTAMP"]/1000)
				curr_time = datetime.utcfromtimestamp(row["TIMESTAMP"]/1000)
				'''
				print("found crash {}".format(crash_cnt))
				print("Prev time: --> " +prev_time.strftime('%Y-%m-%d %H:%M:%S.%f'))
				print("Curr time: --> " +curr_time.strftime('%Y-%m-%d %H:%M:%S.%f'))
				print("seconds: ", time_diff, "distance: ", curr_dist)
				print("Prev row: ", prev_crash_x, prev_crash_y, prev_crash_id_1, prev_crash_id_2)
				print("Curr row: ", curr_x, curr_y, curr_col_id_1, curr_col_id_2)
				'''

				return 1, row
		return 0, prev_crash

	# adds column with 1 indicating new crash occuring at reading, 0 if no crash
	# --> uses __detection_fn as a heuristic for detecting distinct crash
	def __generate_labels(self):
		crash_labels = np.zeros(self.num_rows)
		crash_cnt = 0
		prev_crash = None
		crash_times, crash_x_coords, crash_y_coords = [], [], []
		for index, row in self.df.iterrows():
			if index == 0:
				continue
			if index % 10000 == 0:
				print(index)

			label, prev_crash = self.__detection_fn(row, prev_crash, crash_cnt)
			crash_labels[index] = label
			crash_cnt += label

			if label == 1:
				crash_times.append(datetime.utcfromtimestamp(prev_crash['TIMESTAMP']/1000))
				crash_x_coords.append(prev_crash['POSITION_X'])
				crash_y_coords.append(prev_crash['POSITION_Y'])

		print("found {} crashes".format(crash_cnt))
		for i in range(len(crash_times)):
			print("Crash {}".format(i))
			print("Time {}".format(crash_times[i]))
			print("Coordinates {},{}".format(crash_x_coords[i], crash_y_coords[i]))
			print('')

		return crash_labels

	# generates a column with 1 indicated if a crash occurs WITHIN pred_window_secs of the current reading
	# in the future (these will then be used as the labels in our sequences)
	def __generate_sequence_labels(self, pred_window_secs):
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
			print("Generating crash sequences of {} seconds".format(prev_window_secs))
			crash_label = np.zeros(self.num_rows)
			crash_cnt = 0
			prev_crash = None
			crash_labels = self.__generate_labels()
			self.df['crash_label'] = crash_labels
			# self.df.to_pickle(self.user+"_segmented_crashes.pkl")

		print(datetime.utcfromtimestamp(self.df["TIMESTAMP"][0]/1000))
		print(datetime.utcfromtimestamp(self.df["TIMESTAMP"][self.num_rows-1]/1000))
		print(np.sum(self.df['crash_label']))
		# now that we have detected crashes, create sequence labels
		if load:
			self.df = pd.read_pickle(self.user+"_sequence_labels.pkl")
		else:
			sequences = self.__generate_sequence_labels(pred_window_secs)
			self.df['crash_within_pred_window'] = sequences
			# self.df.to_pickle(self.user+"_sequence_labels.pkl")
		return 1

	def generate_sequences(self, sequence_window_secs=5, crash_window=(5, 10), down_sampling=None):
		def are_sequences_overlapping(sequence1, sequence2):
			if (sequence1[0] < sequence2[0] and sequence1[1] < sequence2[0]) or (sequence2[0] < sequence1[0] and sequence2[1] < sequence1[0]):
				return False
			return True
		def does_sequence_overlap_crash_sequence(sequence, crash_sequences):
			for s in crash_sequences:
				if are_sequences_overlapping(s, sequence): return True
			return False

		crash_indices = self.df.index[self.df['crash_label'] == 1].tolist()
		crash_window_start, crash_window_end = crash_window

		# if the crash is early enough that we can't generate a sequence before it,
		# we'll just go ahead and discard it...
		max_time_before_crash = crash_window_end * READING_HZ
		crash_indices = [idx for idx in crash_indices if idx - max_time_before_crash - sequence_window_secs * READING_HZ > 0]
		print('Generating crash sequences...'.format(len(crash_indices)))
		total_time_sequence_length = sequence_window_secs + crash_window_end


		# for each crash, we generate a timestep that will represent the end of our sequence
		# that timestep will be in the crash_window
		# such that we have sequences of len sequence_window_secs where the crash happens within
		# (crash_window_start, crash_window_end) time from now
		random_time_vals = [np.random.random_integers(low=crash_window_start*READING_HZ, high=crash_window_end*READING_HZ) for i in range(len(crash_indices))]

		# tuples of (sequence_start, sequence_end, crash_index)
		# we include the third to make it easier to get our negative examples
		crash_sequence_indices = [(crash_indices[i] - random_time_vals[i] - sequence_window_secs * READING_HZ, crash_indices[i] - random_time_vals[i]) for i in range(len(crash_indices))]
		crash_extended_sequence_indices = [(crash_indices[i] - random_time_vals[i] - sequence_window_secs * READING_HZ, crash_indices[i]) for i in range(len(crash_indices))]

		no_crash_sequence_indices = []
		num_possible_no_crash_sequences, num_actual_no_crash_sequences = 0, 0
		for i in range(0, self.num_rows, total_time_sequence_length * READING_HZ):
			num_possible_no_crash_sequences += 1
			no_crash_possible_extended_sequence = (i, i + total_time_sequence_length * READING_HZ)
			if does_sequence_overlap_crash_sequence(no_crash_possible_extended_sequence, crash_extended_sequence_indices):
				continue
			num_actual_no_crash_sequences += 1

			start, end = i, i + sequence_window_secs * READING_HZ
			sequence = (start, end)
			if end >= self.num_rows: continue
			no_crash_sequence_indices.append(sequence)

		num_total_sequences = len(crash_sequence_indices) + len(no_crash_sequence_indices)
		print('Found {} total sequences, {} crash sequences and {} no crash sequences'.format(num_total_sequences, len(crash_sequence_indices), len(no_crash_sequence_indices)))
		sequence_length = sequence_window_secs * READING_HZ


		self.df = self.df.drop(['crash_label', 'crash_within_pred_window'], axis=1)
		num_input_cols = len(self.df.columns.values)
		X, Y = np.ndarray((num_total_sequences, sequence_length, num_input_cols)), np.zeros(num_total_sequences)
		for i, (start_seq, end_seq) in enumerate(crash_sequence_indices):
			sequence = self.df.loc[start_seq:end_seq-1].values
			X[i] = sequence
		for i, (start_seq, end_seq) in enumerate(no_crash_sequence_indices):
			sequence = self.df.loc[start_seq:end_seq-1].values
			X[i + len(crash_sequence_indices)] = sequence

		Y[:len(crash_sequence_indices)] += 1
		X = torch.from_numpy(X)
		Y = torch.from_numpy(Y)
		print(X.size(), Y.size())
		print('')

		return X, Y

		# for sequence_start, sequence_end, crash_index in crash_sequence_indices:
		# sequence_length = prev_window_secs*READING_HZ
		# num_sequences = self.num_rows-sequence_length-1
		# num_input_cols = len(self.df.columns.values)
		# X, Y = np.ndarray((num_sequences, sequence_length, num_input_cols)), np.zeros(num_sequences)
		# for seq_idx in range(num_sequences):
		# 	if seq_idx % 10000 == 0:
		# 		print(seq_idx)
		# 	'''
		# 	print(len(self.df.loc[seq_idx:seq_idx+sequence_length-1].values))
		# 	print(self.df.loc[seq_idx:seq_idx+sequence_length-1].values)
		# 	print(self.df.loc[seq_idx+sequence_length]["TIMESTAMP"])
		# 	'''
		# 	X[seq_idx] = self.df.loc[seq_idx:seq_idx+sequence_length-1].values
		# 	Y[seq_idx] = self.df.loc[seq_idx+sequence_length]['crash_within_pred_window']
		# print(len(X), len(X[0]))
		# print(X[0])
		# print(len(Y))
		# print(Y[0])
		# self.X = X
		# self.Y = Y

		# print(self.X.shape)
		# print(self.Y.shape)




# dl = DriverData(DATA, load=True)
# # plot_course(dl.df)
# dl.segment_crashes(load=True)
# dl.generate_sequences()

X_tensors = []
Y_tensors = []

prev_set = None
for f in os.listdir(DATA_DIR):
	print('processing data for {}'.format(f))
	driver = DriverData(f, load=False)

	driver.segment_crashes(load=False)
	X, Y = driver.generate_sequences()
	X_tensors.append(X)
	Y_tensors.append(Y)
	print('')

X_final = torch.cat(X_tensors, dim=0)
Y_final = torch.cat(Y_tensors, dim=0)
print('final shape of X data', X_final.size())
print('final shape of Y data', Y_final.size())
torch.save(X_final, PYTORCH_DATA_DIR + 'data.pt')
torch.save(Y_final, PYTORCH_DATA_DIR + 'labels.pt')


