import time
import pandas as pd

class Timer:
	def __init__(self, label):
		self.label = label
		self.start_time = time.time()

	def stop(self):
		self.end_time = time.time()
		label = self.label + " : " + str(self.end_time - self.start_time)
		print(label)



def pprint(data_frame):
	with pd.option_context('display.max_rows', None, 'display.max_columns', 10):
		pd.set_option('display.width', 1000)
		print data_frame