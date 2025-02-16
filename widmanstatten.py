import tkinter as tk
import numpy as np
from random import randint, choice
from dataclasses import dataclass, field

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
    color: str
    speed: int
    id: str
    growing_right: bool
    growing_left: bool
    limit_right: int
    limit_left: int
    angle: float
    cos: float = field(init=False)
    sin: float = field(init=False)

    def __post_init__(self):
        self.cos = np.cos(self.angle)
        self.sin = np.sin(self.angle)

class AnimationWindow:

    def __init__(self, 
                 width=500, 
                 height=500, 
                 fps=4, 
                 init_crystal_count=10, 
                 mean_crystal_width=10, 
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
        print(distances)
        self.set_growth_limits(distances)
        
        self.render_crystals()
        self.render_centers()
        self.root.after(self.frame_delay, self.render_crystals)
        self.root.mainloop()

    def calc_intersection_distances(self):

        crystal_data = np.array([(crystal.center.x, crystal.center.y, crystal.angle, crystal.cos, crystal.speed) for crystal in self.crystals], dtype=float)
        
        x_coords = np.array(np.meshgrid(crystal_data[:,0], crystal_data[:,0]))
        y_coords = np.array(np.meshgrid(crystal_data[:,1], crystal_data[:,1]))
        tangents = np.tan(np.array(np.meshgrid(crystal_data[:,2], crystal_data[:,2])))
        cosines = np.array(np.meshgrid(crystal_data[:,3], crystal_data[:,3]))
        growth_speeds = crystal_data[:,4]

        eps = 10**-8
        distances_with_asymptotes = (y_coords[0] - y_coords[1] + tangents[0] * (x_coords[0] - x_coords[1])) / (cosines[1] * (tangents[0] - tangents[1]) + eps)
        distances = np.where(tangents[0] == tangents[1], 0, distances_with_asymptotes)

        distances = distances / growth_speeds[...,np.newaxis]
        return distances
    
    def set_growth_limits(self, dist_matrix):

        reached_xsections = np.zeros((self.init_crystal_count, self.init_crystal_count))
        right_limits = np.full(self.init_crystal_count, self.max_length)
        left_limits = np.full(self.init_crystal_count, self.max_length)


        for crystal_index, intersection_distances in enumerate(dist_matrix):
            for intersection_index in np.argsort(intersection_distances)[::-1][:np.sum(intersection_distances > 0)]:
                if self.line_passes_xsection(dist_matrix, reached_xsections, crystal_index, intersection_index):
                    break
                else:
                    right_limits[crystal_index] = abs(intersection_distances[intersection_index])

            for intersection_index in np.argsort(intersection_distances)[:np.sum(intersection_distances < 0)]:
                if self.line_passes_xsection(dist_matrix, reached_xsections, crystal_index, intersection_index):
                    break
                else:
                    left_limits[crystal_index] = abs(intersection_distances[intersection_index])

        print(right_limits)
        print(left_limits)

        for i, crystal in enumerate(self.crystals):
            crystal.limit_right = right_limits[i]*crystal.speed
            crystal.limit_left = left_limits[i]*crystal.speed


    def line_passes_xsection(self, dist_matrix, reached_xsections, main_index, cross_index):
        
        crystal_reaches_xsection = self.line_reaches_xsection(dist_matrix, reached_xsections, main_index, cross_index)
        crystal_arrives_first = np.abs(dist_matrix[main_index, cross_index]) < np.abs(dist_matrix[cross_index, main_index])

        # To prevent issues with deeper recursive calls writing into reached_xsections before higher calls finish, 
        # self.line_reaches_xsection below should be only called if crystal_reaches_xsection = True and crystal_arrives_first = False
        # This is done by compiler prioritization and the call shouldn't be refactored above like the other 2 expressions
        return crystal_reaches_xsection and (crystal_arrives_first or not self.line_reaches_xsection(dist_matrix, reached_xsections, cross_index, main_index))
        

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

                if reached_xsections[main_index, xsection] != 1 and not self.line_passes_xsection(dist_matrix, reached_xsections, main_index, xsection):
                    reached_xsections[main_index][dist_matrix[main_index] > dist_matrix[main_index, xsection]] = -1
                    return False

        if main_dist < 0:
            for xsection in np.argsort(np.where(
                    np.logical_and(dist_matrix[main_index] < 0, dist_matrix[main_index] > main_dist), dist_matrix[main_index], -np.inf))[::-1][:sum(np.logical_and(dist_matrix[main_index] < 0, dist_matrix[main_index] > main_dist))]:

                if reached_xsections[main_index, xsection] != 1 and not self.line_passes_xsection(dist_matrix, reached_xsections, main_index, xsection):
                    reached_xsections[main_index][dist_matrix[main_index] < dist_matrix[main_index, xsection]] = -1
                    return False
                
        reached_xsections[main_index, cross_index] = 1
        return True
    
    def render_crystals(self):

        for crystal in self.crystals:
            if crystal.growing_right or crystal.growing_left:
                self.redraw_crystal(crystal)
                self.extend_crystal(crystal)
                
        self.root.after(self.frame_delay, self.render_crystals)

    def extend_crystal(self, crystal: Crystal):

        if crystal.growing_right:
            self.extend_crystal_right(crystal)
        if crystal.growing_left:
            self.extend_crystal_left(crystal)

    def extend_crystal_right(self, crystal: Crystal):

        crystal.length_right += crystal.speed
        if crystal.length_right >= crystal.limit_right:
            crystal.growing_right = False

    def extend_crystal_left(self, crystal: Crystal):

        crystal.length_left += crystal.speed
        if crystal.length_left >= crystal.limit_left:
            crystal.growing_left = False

    def redraw_crystal(self, crystal: Crystal):

        p1, p2, p3, p4 = self.calc_crystal_corners(crystal)
        self.canvas.delete(crystal.id)
        self.canvas.create_polygon(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y, p4.x, p4.y, fill=crystal.color, tags=crystal.id)

    def calc_crystal_corners(self, crystal: Crystal):

        half_width = crystal.width // 2

        p1 = Point(crystal.center.x + crystal.length_right * crystal.cos - half_width * crystal.sin, crystal.center.y - crystal.length_right * crystal.sin - half_width * crystal.cos)
        p2 = Point(crystal.center.x + crystal.length_right * crystal.cos + half_width * crystal.sin, crystal.center.y - crystal.length_right * crystal.sin + half_width * crystal.cos)
        p3 = Point(crystal.center.x - crystal.length_left * crystal.cos + half_width * crystal.sin, crystal.center.y + crystal.length_left * crystal.sin + half_width * crystal.cos)
        p4 = Point(crystal.center.x - crystal.length_left * crystal.cos - half_width * crystal.sin, crystal.center.y + crystal.length_left * crystal.sin - half_width * crystal.cos)

        return p1, p2, p3, p4

    def render_centers(self, radius: int = 10, color: str = "red"):

        for crystal in self.crystals:
            self.canvas.create_oval(crystal.center.x - radius, crystal.center.y - radius, crystal.center.x + radius, crystal.center.y + radius, fill=color)


        

    
master = AnimationWindow()
master.create_window()
