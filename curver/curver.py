"""
Main Curver file:
Curve reconstruction from noisy data
"""

import numpy as np
import statsmodels.api as sm

"""
NB in general first column of data will by X 
and second column Y, though just convention
"""

class Curver:
	""" Class for curve reconstruction """

	def __init__(self, points=None):
		self.points = points
		self.W = None # weight matrix


	def from_csv(self, filename):
		""" Load point cloud from filename csv """
		points = np.genfromtxt(filename, delimiter=",")
		assert points.shape[1] == 2

		self.N = points.shape[0]
		self.points = points


	def reconstruct(self):
		""" reconstruction routine """
		self.weight_matrix()


	def weight_matrix(self, H=2):
		R = np.zeros(shape=(self.N,self.N))
		"""
		Fill out bottom left corner of W then add to transpose
		"""

		for i in xrange(1, self.N):
			for j in xrange(0, i):
				x = self.points[i,:] - self.points[j,:]
				R[i,j] = x.dot(x)

		R = R + R.T
		self.R = R
		w = 2 / H**3 * np.power(R,3) - 3 / H**2 * np.power(R,2) + 1
		w[R > H] = 0
		np.fill_diagonal(w, 1)

		self.W = w

	def do_first_regression(self, point_index):
		""" Performs the initial regression step
		Minimises D_l for the point specified by point point_index
		"""
		weights = self.W[point_index,:]
		points = self.points[weights > 0,:]

		weights = weights[weights > 0]
		
		assert points.shape[0] == len(weights), "%d rows of points isn't equal to %d length of weights" % (points.shape[0],len(weights))

		wls_model = sm.WLS(points[:,1], points[:,0], weights = weights)
		results = wls_model.fit()
		return results.params








