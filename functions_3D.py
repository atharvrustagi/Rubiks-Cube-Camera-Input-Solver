import numpy as np
from cube_functions import *

Zv = 1000
f = 960/np.tan(np.pi/4)

def project_surfaces(cube, alpha, beta, pos):
	h = (cube[..., 0]**2 + cube[..., 2]**2)**0.5
	a = np.arctan(cube[..., 2]/(cube[..., 0] + 1e-8)) - alpha
	c = np.where(cube[..., 0]>=0, 1, -1)
	cube[..., 0] = c*h*np.cos(a)
	cube[..., 2] = c*h*np.sin(a)

	h = (cube[..., 1]**2 + cube[..., 2]**2)**0.5
	a = np.arctan(cube[..., 2]/(cube[..., 1] + 1e-8)) - beta
	c = np.where(cube[..., 1]>=0, 1, -1)
	cube[..., 1] = c*h*np.cos(a)
	cube[..., 2] = c*h*np.sin(a)

	z = np.mean(cube[..., 2], axis=3).reshape(54)
	ret = (f*cube[..., :2]/(Zv+cube[..., 2:])).reshape(54, 4, 2)
	ret[..., 0] += pos[0]
	ret[..., 1] += pos[1]
	return ret, z

# counters
NTURN = 125			# must be a power of 5 to work properly
ncounter = 0
mcounter = 0

def play_moves(solution_list, cube, colors):
	global ncounter, mcounter
	move = solution_list[mcounter]

	if move.find('2')==1:
		move = algo_move[move[0]]	# converting to number
		if ncounter<NTURN*2:
			moves_animate[move](np.pi/2/NTURN, cube)
			ncounter += 1
		else:
			moves_animate[move](-np.pi, cube)
			moves[move](colors)
			moves[move](colors)
			ncounter = 0
			mcounter += 1
	else:
		move = algo_move[move]			# converting to number
		if ncounter<NTURN:
			moves_animate[move%6]((-1 if move>=6 else 1)*np.pi/2/NTURN, cube)
			ncounter += 1
		else:
			moves_animate[move%6]((1 if move>=6 else -1)*np.pi/2, cube)
			moves[move](colors)
			ncounter = 0
			mcounter += 1

	return mcounter < len(solution_list)
