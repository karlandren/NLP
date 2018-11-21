from Key import Key
import math
import sys
import numpy as np
import codecs
import argparse
import json
import requests

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.
"""

class ViterbiTrigramDecoder(object):
	"""
	This class implements Viterbi decoding using trigram probabilities in order
	to correct keystroke errors.
	"""
	def init_a(self, filename):
		"""
		Reads the trigram probabilities (the 'A' matrix) from a file.
		"""
		with codecs.open(filename, 'r', 'utf-8') as f:
			for line in f:
				i, j, k, d = [func(x) for func, x in zip([int, int, int, float], line.strip().split(' '))]
				self.a[i][j][k] = d


	# ------------------------------------------------------


	def init_b(self):
		"""
		Initializes the observation probabilities (the 'B' matrix).
		"""
		for i in range(Key.NUMBER_OF_CHARS):
			cs = Key.neighbour[i]

			# Initialize all log-probabilities to some small value.
			for j in range(Key.NUMBER_OF_CHARS):
				self.b[i][j] = -float("inf")

			# All neighbouring keys are assigned the probability 0.1
			for j in range(len(cs)):
				self.b[i][Key.char_to_index(cs[j])] = math.log( 0.1 )

			# The remainder of the probability mass is given to the correct key.
			self.b[i][i] = np.log((10 - len(cs))/10.0)


	# ------------------------------------------------------



	def viterbi(self, s):
		"""
		Performs the Viterbi decoding and returns the most likely
		string.
		"""
		# First turn chars to integers, so that 'a' is represented by 0,
		# 'b' by 1, and so on.
		index = [Key.char_to_index(x) for x in s]

		# The Viterbi matrices
		self.v = np.zeros((len(s), Key.NUMBER_OF_CHARS, Key.NUMBER_OF_CHARS), dtype='double')
		self.v[:,:,:] = -float("inf")
		self.backptr = np.zeros((len(s), Key.NUMBER_OF_CHARS, Key.NUMBER_OF_CHARS), dtype='int')

		# Initialization

		# YOUR CODE HERE

		self.v[0,:,:] = self.a[Key.START_END,Key.START_END,:] + self.b[index[0],:]
		# v[0,Key.START_END,:] -> prob of all states at obs 0 for succeding state key.START_END
		# prob all states 0->26 transition | 26,26
		# prob all states 0->26 emitting 26 -> only space can emit 26

		self.backptr[0,:,:] = Key.START_END

		# a[i,j,k] -> prob of going to state k given preceding was j and the one before was i, P(i,j,k) = P(k|i,j)
		# b[i,k] -> prob of state k emitting oi
		# v[t,i,k] -> prob of being in state k, passing through most probable succeding state i after first t obs

		# Induction step

		# YOUR CODE HERE

		n_obs = len(s)
		n_states = len(self.a)

		for obs_idx in range(1, n_obs):
			for state_idx in range(n_states):
				for prec_idx in range(n_states):
					
					t_probs = self.v[obs_idx-1,:,prec_idx] + self.a[:,prec_idx,state_idx]
					self.v[obs_idx,prec_idx,state_idx] = max(t_probs) + self.b[index[obs_idx],state_idx]
					self.backptr[obs_idx,prec_idx,state_idx] = np.argmax(t_probs)


		c = len(self.v)-1
		this = Key.START_END
		next_state = Key.START_END
		after_next = Key.START_END
		msg = ''
		while c >= -1:
			msg = Key.index_to_char(this) + msg
			this = next_state
			next_state = after_next
			after_next = self.backptr[c,next_state,this]
			c -= 1

		# Finally return the result

		# REPLACE THE LINE BELOW WITH YOUR CODE

		return msg.strip()



	# ------------------------------------------------------



	def __init__(self, filename=None):
		"""
		Constructor: Initializes the A and B matrices.
		"""
		# The trellis used for Viterbi decoding. The first index is the time step.
		self.v = None

		# The trigram stats.
		self.a = np.zeros((Key.NUMBER_OF_CHARS, Key.NUMBER_OF_CHARS, Key.NUMBER_OF_CHARS), dtype='double')

		# The observation matrix.
		self.b = np.zeros((Key.NUMBER_OF_CHARS, Key.NUMBER_OF_CHARS), dtype='double')

		# Pointers to retrieve the topmost hypothesis.
		backptr = None

		if filename: self.init_a(filename)
		self.init_b()



	# ------------------------------------------------------

def main():

	parser = argparse.ArgumentParser(description='ViterbiTrigram decoder')
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument('--file', '-f', type=str, help='decode the contents of a file')
	group.add_argument('--string', '-s', type=str, help='decode a string')
	parser.add_argument('--probs', '-p', type=str,  required=True, help='trigram probabilities file')
	parser.add_argument('--check', action='store_true', help='check if your answer is correct')

	arguments = parser.parse_args()

	if arguments.file:
		with codecs.open(arguments.file, 'r', 'utf-8') as f:
			s1 = f.read().replace('\n', '')
	elif arguments.string:
		s1 = arguments.string

	# Give the filename of the trigram probabilities as a command line argument
	d = ViterbiTrigramDecoder(arguments.probs)

	# Append two extra "END" symbols to the input string, to indicate end of sentence.
	result = d.viterbi(s1 + Key.index_to_char(Key.START_END) + Key.index_to_char(Key.START_END))

	if arguments.check:
		payload = json.dumps({
			'a': d.a.tolist(),
			'string': s1,
			'result': result
		})
		response = requests.post(
			'https://language-engineering.herokuapp.com/lab3_trigram',
			data=payload,
			headers={'content-type': 'application/json'}
		)
		response_data = response.json()
		if response_data['correct']:
			print(result)
			print('Success! Your results are correct')
		else:
			print('Your results:')
			print(result)
			print('Your answer is {0:.0f}% similar to the servers'.format(response_data['result'] * 100))
	else:
		print(result)

if __name__ == "__main__":
	main()
