import tkinter as tk
import numpy as np
from math import cos, sin, pi
from random import randint, choice
from dataclasses import dataclass

np.set_printoptions(precision=2)

@dataclass(slots=True)
class Point:
    x: int
    y: int

@dataclass(slots=True)
class Crystal:
    center: Point
    length_right: int
    length_left: int
    width: int
    angle: float
    color: str
    speed: int
    id: str
    growing_right: bool
    growing_left: bool
    limit_right: int
    limit_left: int

class AnimationWindow:

    def __init__(self, 
                 width=500, 
                 height=500, 
                 fps=2, 
                 init_crystal_count=10, 
                 mean_crystal_width=20, 
                 mean_crystal_speed=5, 
                 orientation_angles=[-1, 0, 1], 
                 background="steelblue4"):
        
        self.width = width
        self.height = height
        self.margin = self.width // 10
        self.frame_delay = 1000//fps
        self.init_crystal_count = init_crystal_count
        self.init_crystal_length = 1
        self.mean_crystal_width = mean_crystal_width
        self.mean_crystal_speed = mean_crystal_speed
        self.max_length = np.sqrt(self.width**2 + self.height**2)
        self.orientation_angles = orientation_angles # 60 degrees = approx 1 rad

        self.crystals = [Crystal(center = Point(randint(self.margin, self.width - self.margin), randint(self.margin, self.height-self.margin)), 
                                 length_right = self.init_crystal_length,
                                 length_left = self.init_crystal_length,
                                 width = randint(self.mean_crystal_width // 2, self.mean_crystal_width * 3 // 2), 
                                 angle = choice(self.orientation_angles),
                                 color = "gray" + str(id * 100 // init_crystal_count),
                                 speed = randint(self.mean_crystal_speed * 3 // 4, self.mean_crystal_speed * 5 // 4),
                                 id = str(id),
                                 growing_right = True,
                                 growing_left = True,
                                 limit_right = self.max_length,
                                 limit_left = self.max_length)
                                 for id in range(self.init_crystal_count)]

        self.root = tk.Tk()
        self.root.title = "Widmastatten patern growth animation"
        self.root.geometry(f"{self.width}x{self.height}")

        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg=background)
        self.canvas.pack()

    def create_window(self):
                
        distances = self.calc_intersection_distances()
        self.set_length_limits(distances)
        
        self.render_crystals()
        self.render_centers()

        self.root.after(self.frame_delay, self.render_crystals)

        self.root.mainloop()

    def calc_crystal_corners(self, crystal: Crystal):

        c = cos(crystal.angle)
        s = sin(crystal.angle)

        half_width = crystal.width // 2

        p1 = Point(crystal.center.x + crystal.length_right*c - half_width*s, crystal.center.y - crystal.length_right*s - half_width*c)
        p2 = Point(crystal.center.x + crystal.length_right*c + half_width*s, crystal.center.y - crystal.length_right*s + half_width*c)
        p3 = Point(crystal.center.x - crystal.length_left*c + half_width*s, crystal.center.y + crystal.length_left*s + half_width*c)
        p4 = Point(crystal.center.x - crystal.length_left*c - half_width*s, crystal.center.y + crystal.length_left*s - half_width*c)

        return p1, p2, p3, p4
    
    def calc_intersection_distances(self):

        data = np.array([(crystal.center.x, crystal.center.y, crystal.angle, crystal.speed) for crystal in self.crystals], dtype=float)
        
        x_coords = np.array(np.meshgrid(data[:,0], data[:,0]))
        y_coords = np.array(np.meshgrid(data[:,1], data[:,1]))
        growth_speeds = data[:,3]
        cosines = np.cos(np.array(np.meshgrid(data[:,2], data[:,2])))
        tangents = np.tan(np.array(np.meshgrid(data[:,2], data[:,2])))

        distances = (y_coords[0] - y_coords[1] + tangents[0] * (x_coords[0] - x_coords[1])) / (cosines[1] * (tangents[0] - tangents[1]) + 10**-8)
        distances = np.where(tangents[0] == tangents[1], 0, distances)

        distances = distances / growth_speeds[...,np.newaxis]
        return distances

    def set_length_limits(self, distances):

        reached_xsections = np.zeros((self.init_crystal_count, self.init_crystal_count))
        right_limits = np.full(self.init_crystal_count, self.max_length)
        left_limits = np.full(self.init_crystal_count, self.max_length)


        for main_index, line in enumerate(distances):
            for cross_index in np.argsort(line)[::-1][:np.sum(line > 0)]:
                if self.line_passes_xsection(distances, reached_xsections, main_index, cross_index):
                    break
                else:
                    right_limits[main_index] = abs(line[cross_index])
            for cross_index in np.argsort(line)[:np.sum(line < 0)]:
                if self.line_passes_xsection(distances, reached_xsections, main_index, cross_index):
                    break
                else:
                    left_limits[main_index] = abs(line[cross_index])

        for i in range(self.init_crystal_count):
            self.crystals[i].limit_right = right_limits[i]*self.crystals[i].speed
            self.crystals[i].limit_left = left_limits[i]*self.crystals[i].speed





    def line_passes_xsection(self, dist_matrix, reached_xsections, main_index, cross_index):
        if (self.line_reaches_xsection(dist_matrix, reached_xsections, main_index, cross_index) 
            and (np.abs(dist_matrix[main_index, cross_index]) < np.abs(dist_matrix[cross_index, main_index]) 
                 or not self.line_reaches_xsection(dist_matrix, reached_xsections, cross_index, main_index))):
            return True
        return False

    def line_reaches_xsection(self, dist_matrix, reached_xsections, main_index, cross_index):


        if reached_xsections[main_index, cross_index] != 0:
            return reached_xsections[main_index, cross_index] == 1
        main_dist = dist_matrix[main_index, cross_index]

        if main_dist == 0:
            reached_xsections[main_index, cross_index] = 1
            return True
        
        if main_dist > 0:
            for xsection in np.argsort(np.where(
                    np.logical_and(dist_matrix[main_index] > 0, dist_matrix[main_index] < main_dist), dist_matrix[main_index], np.inf))[:sum(np.logical_and(dist_matrix[main_index] > 0, dist_matrix[main_index] < main_dist))]:
                
                if reached_xsections[main_index, xsection] == 1:
                    continue
                elif reached_xsections[main_index, xsection] == -1:
                    raise(Exception("fuck you know what"))
                if self.line_passes_xsection(dist_matrix, reached_xsections, main_index, xsection):
                    reached_xsections[main_index, xsection] = 1
                else:
                    reached_xsections[main_index][dist_matrix[main_index] > dist_matrix[main_index, xsection]] = -1
                    return False

        if main_dist < 0:
            for xsection in np.argsort(np.where(
                    np.logical_and(dist_matrix[main_index] < 0, dist_matrix[main_index] > main_dist), dist_matrix[main_index], -np.inf))[::-1][:sum(np.logical_and(dist_matrix[main_index] < 0, dist_matrix[main_index] > main_dist))]:
                
                if reached_xsections[main_index, xsection] == 1:
                    continue
                elif reached_xsections[main_index, xsection] == -1:
                    raise(Exception("fuck you know what"))
                if self.line_passes_xsection(dist_matrix, reached_xsections, main_index, xsection):
                    reached_xsections[main_index, xsection] = 1
                else:
                    reached_xsections[main_index][dist_matrix[main_index] < dist_matrix[main_index, xsection]] = -1
                    return False
                
        reached_xsections[main_index, cross_index] = 1
        return True


    def redraw_crystal(self, crystal: Crystal):

        p1, p2, p3, p4 = self.calc_crystal_corners(crystal)

        self.canvas.delete(crystal.id)
        self.canvas.create_polygon(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y, p4.x, p4.y, fill=crystal.color, tags=crystal.id)

    def render_crystals(self):
        for i in range(len(self.crystals)):
            if self.crystals[i].growing_right or self.crystals[i].growing_left:
                self.redraw_crystal(self.crystals[i])
                if self.crystals[i].growing_right:
                    self.crystals[i].length_right += self.crystals[i].speed
                    if self.crystals[i].length_right >= self.crystals[i].limit_right:
                        self.crystals[i].growing_right = False
                if self.crystals[i].growing_left:
                    self.crystals[i].length_left += self.crystals[i].speed
                    if self.crystals[i].length_left >= self.crystals[i].limit_left:
                        self.crystals[i].growing_left = False
        
        self.root.after(self.frame_delay, self.render_crystals)

    def render_centers(self):
        for crystal in self.crystals:
            self.canvas.create_oval(crystal.center.x - 10, crystal.center.y - 10, crystal.center.x + 10, crystal.center.y + 10, fill="red")


        

    
master = AnimationWindow()
master.create_window()
