import pygame as pg
import cv2
from utils import *
from time import perf_counter as pf
import numpy as np
import os

# pygame parameters
WINSIZE = (1920, 1080)		
win = pg.display.set_mode()		# blank means full screen
clock = pg.time.Clock()

# cv2 parameters
cam = cv2.VideoCapture(0)
cam.set(3, 960)
cam.set(4, 720)


# draw function
def draw(cam_img, flat_cube_colors, cube_params, lens_image, state, soln=None, solved=False):
	win.fill(clrs['w'])
	win.blit(lens_image, (1440-25, 160))
	blit_text(win, state, soln, solved)
	win.blit(cam_img, (0, 0))
	draw_flat_cube(flat_cube_colors, win, state)
	draw_cube(cube_params, win)
	pg.display.update()


# mainloop
def main(run):

	state = 1
	alpha, beta = 1e-3+np.pi, 1e-3-np.pi/8+np.pi/2
	dalpha, dbeta = 0, 0
	nturn_angle = 50
	cube_colors = init_colors()
	cube = create_cube(50)		# cube with side = 50
	pos = (int(1920*3/4), 360)
	lens_image = pg.image.load('lens.png')
	lens_image = pg.transform.scale(lens_image, (50, 50))
	frame_read = 0
	solution = None
	solved = False
	solution_list = None
	start_playing = False

	while run:
		t = pf()

		keys = pg.key.get_pressed()
		dalpha, dbeta = d_angles(keys, nturn_angle, dalpha, dbeta)

		# reading camera input every 3 frames
		if frame_read == 0:
			frame_read = 3
			src, img = cam.read()
			img = cv2.flip(img, 1)
			if state<=6:
				col, img = find_colors(img)

			# transforming to pygame image
			img = get_pg_image(img)

		frame_read -= 1

		for e in pg.event.get():
			if e.type==pg.QUIT:
				run = False
				os.remove('temp.jpg')
			if e.type==pg.MOUSEBUTTONDOWN or keys[pg.K_o] or keys[pg.K_RETURN]:
				if state<7:
					change_cube_params(cube_colors, col, state)
					for _ in range(nturn_angle):
						alpha, beta = change_angle(state, nturn_angle, alpha, beta)
						cube_params = {'cube':cube, 'colors':cube_colors, 'a':alpha+dalpha, 'b':beta+dbeta, 'pos':pos}
						draw(img, col, cube_params, lens_image, state)
					state += 1
				elif solution_list:
					start_playing = True

		if state==7 and not solved:
			# solution, time = solve_cube(cube_colors)
			try:
				solution, time = solve_cube(cube_colors)
				print(f"\n\nSolution took {time} seconds and {solution.count(' ')+1} moves")
				solution_list = solution.split(' ')
				print(solution_list)
				solved = True
			except Exception as e:
				solution = str(e)
				state += 1

		if start_playing:
			start_playing = play_moves(solution_list, cube, cube_colors)

		cube_params = {'cube':cube, 'colors':cube_colors, 'a':alpha+dalpha, 'b':beta+dbeta, 'pos':pos}

		draw(img, col, cube_params, lens_image, state, solution, solved)

		# print(f"FPS: {round(1/(pf()-t))}")


if __name__ == "__main__":
	main(True)

