import tkinter as tk
import numpy as np
from random import randint, choice
from dataclasses import dataclass, field
from enum import Enum

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

class IntersectionState(Enum):
    NOTREACHED = -1
    UNKNOWN = 0
    REACHED = 1

class AnimationWindow:

    def __init__(self, 
                 width=500, 
                 height=500, 
                 fps=5, 
                 init_crystal_count=10,
                 mean_crystal_width=10, 
                 mean_crystal_speed=5, 
                 orientation_angles=[-1., 0, 1.],
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
        self.canvas.bind("<Button-1>", self.add_crystal)

    def create_window(self):

        distances = self.calc_intersection_distances()
        self.set_growth_limits(distances)
        
        self.render_crystals()
        #self.render_centers()
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

        intersection_states = np.full((len(self.crystals), len(self.crystals)), IntersectionState.UNKNOWN)
        right_limits = np.full(len(self.crystals), self.max_length)
        left_limits = np.full(len(self.crystals), self.max_length)


        for crystal_index, intersection_distances in enumerate(dist_matrix):
            for intersection_index in np.argsort(intersection_distances)[::-1][:np.sum(intersection_distances > 0)]:
                if self.crystal_passes_xsection(dist_matrix, intersection_states, crystal_index, intersection_index):
                    break
                else:
                    right_limits[crystal_index] = abs(intersection_distances[intersection_index])

            for intersection_index in np.argsort(intersection_distances)[:np.sum(intersection_distances < 0)]:
                if self.crystal_passes_xsection(dist_matrix, intersection_states, crystal_index, intersection_index):
                    break
                else:
                    left_limits[crystal_index] = abs(intersection_distances[intersection_index])
        
        self.intersection_states_cache = intersection_states

        for i, crystal in enumerate(self.crystals):
            crystal.limit_right = right_limits[i]*crystal.speed
            crystal.limit_left = left_limits[i]*crystal.speed

    def crystal_passes_xsection(self, dist_matrix, intersection_states, crystal_index, obstacle_index):
        
        crystal_reaches_xsection = self.crystal_reaches_xsection(dist_matrix, intersection_states, crystal_index, obstacle_index)
        crystal_arrives_first = np.abs(dist_matrix[crystal_index, obstacle_index]) < np.abs(dist_matrix[obstacle_index, crystal_index])

        # To prevent issues with deeper recursive calls writing into reached_xsections before higher calls finish, 
        # self.crystal_reaches_xsection below should be only called if crystal_reaches_xsection = True and crystal_arrives_first = False
        # This is done by compiler prioritization and the call shouldn't be refactored above like the other 2 expressions
        return crystal_reaches_xsection and (crystal_arrives_first or not self.crystal_reaches_xsection(dist_matrix, intersection_states, obstacle_index, crystal_index))
        

    def crystal_reaches_xsection(self, dist_matrix, intersection_states, crystal_index, obstacle_index):

        if intersection_states[crystal_index, obstacle_index] != IntersectionState.UNKNOWN:
            return intersection_states[crystal_index, obstacle_index] == IntersectionState.REACHED
        main_dist = dist_matrix[crystal_index, obstacle_index]

        if main_dist == 0:
            intersection_states[crystal_index, obstacle_index] = IntersectionState.REACHED
            return True

        sign = int(np.sign(main_dist))
        for crossing in np.argsort(np.where(
                np.logical_and(np.sign(dist_matrix[crystal_index]) == sign, dist_matrix[crystal_index]*sign < main_dist*sign), 
                dist_matrix[crystal_index], np.inf*sign))[::sign][:sum(
                    np.logical_and(np.sign(dist_matrix[crystal_index]) == sign, dist_matrix[crystal_index]*sign < main_dist*sign))]:

            if (intersection_states[crystal_index, crossing] != IntersectionState.REACHED 
                and not self.crystal_passes_xsection(dist_matrix, intersection_states, crystal_index, crossing)):
                intersection_states[crystal_index][dist_matrix[crystal_index]*sign > dist_matrix[crystal_index, crossing]*sign] = IntersectionState.NOTREACHED
                return False
                
        intersection_states[crystal_index, obstacle_index] = IntersectionState.REACHED
        return True
    
    def add_crystal(self, event):
        center = Point(event.x, event.y)
        self.crystals.append(Crystal(
            center = center,
            length_right = self.init_crystal_length,
            length_left = self.init_crystal_length,
            width = randint(self.mean_crystal_width // 2, self.mean_crystal_width * 3 // 2), 
            angle = choice(self.orientation_angles),
            color = "yellow",
            speed = randint(self.mean_crystal_speed * 3 // 4, self.mean_crystal_speed * 5 // 4),
            id = str(int(self.crystals[-1].id) + 1),
            growing_right = True,
            growing_left = True,
            limit_right = self.max_length,
            limit_left = self.max_length)
        )

        distances = self.calc_intersection_distances() # yes, this could be made more efficient by writing another iterative distance calculator
        self.set_last_crystal_growth_limit(distances)

    def set_last_crystal_growth_limit(self, dist_matrix):
        

        intersection_states = np.full((len(self.crystals), len(self.crystals)), IntersectionState.UNKNOWN)
        intersection_states[:self.intersection_states_cache.shape[0], :self.intersection_states_cache.shape[1]] = self.intersection_states_cache

        right_limit = self.max_length
        left_limit = self.max_length

        crystal_index = -1
        intersection_distances = dist_matrix[crystal_index,:]

        for intersection_index in np.argsort(intersection_distances)[-np.sum(intersection_distances > 0):]:
            intersection_states[crystal_index, intersection_index] = IntersectionState.REACHED
            if self.crystal_reaches_xsection(dist_matrix, intersection_states, intersection_index, crystal_index):
                right_limit = abs(intersection_distances[intersection_index])
                sign = np.sign(dist_matrix[crystal_index, intersection_index])
                intersection_states[crystal_index][dist_matrix[crystal_index]*sign > dist_matrix[crystal_index, intersection_index]*sign] = IntersectionState.NOTREACHED
                break

        for intersection_index in np.argsort(intersection_distances)[:np.sum(intersection_distances < 0)][::-1]:
            intersection_states[crystal_index, intersection_index] = IntersectionState.REACHED
            if self.crystal_reaches_xsection(dist_matrix, intersection_states, intersection_index, crystal_index):
                left_limit = abs(intersection_distances[intersection_index])
                sign = np.sign(dist_matrix[crystal_index, intersection_index])
                intersection_states[crystal_index][dist_matrix[crystal_index]*sign > dist_matrix[crystal_index, intersection_index]*sign] = IntersectionState.NOTREACHED
                break
                

        self.intersection_states_cache = intersection_states

        
        self.crystals[-1].limit_right = right_limit*self.crystals[-1].speed
        self.crystals[-1].limit_left = left_limit*self.crystals[-1].speed

    
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
