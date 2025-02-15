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

class WPattern:

    def __init__(self):
        
        self.w = 500
        self.h = 500
        self.margin = self.w // 10
        self.init_crystal_count = 5

        self.mean_crystal_width = 10
        self.mean_crystal_speed = 1
        self.crystal_half_length = 50
        self.orientation_angles = [-1, 0, 1] # 60 degrees = approx 1 rad

        self.crystals = [Crystal(center = Point(randint(self.margin, self.w - self.margin), randint(self.margin, self.h-self.margin)), 
                                 length_right = self.crystal_half_length, 
                                 length_left = self.crystal_half_length,
                                 width = self.mean_crystal_width + randint(-self.mean_crystal_width // 2, self.mean_crystal_width // 2), 
                                 angle = choice(self.orientation_angles),
                                 color = "gray" + str(id),#str(randint(30, 80)),
                                 speed = self.mean_crystal_speed ,#+ randint(-self.mean_crystal_speed // 2, self.mean_crystal_speed // 2),
                                 id = str(id),
                                 growing_right = True,
                                 growing_left = True)
                                 for id in range(0, self.init_crystal_count*20, 20)]

        self.root = tk.Tk()
        self.root.title = "Widmastatten patern growth animation"
        self.root.geometry(f"{self.w}x{self.h}")

        self.canvas = tk.Canvas(self.root, width=self.w, height=self.h, bg="steelblue4")
        self.canvas.pack()

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

        data = np.array([(crystal.center.x, crystal.center.y, crystal.angle, crystal.speed) for crystal in self.crystals], dtype=int)
        print(data)
        
        x_coords = np.array(np.meshgrid(data[:,0], data[:,0]))
        y_coords = np.array(np.meshgrid(data[:,1], data[:,1]))
        cosines = np.cos(np.array(np.meshgrid(data[:,2], data[:,2]), dtype=float))
        tangents = np.tan(np.array(np.meshgrid(data[:,2], data[:,2]), dtype=float))

        distances = (y_coords[0] - y_coords[1] + tangents[0] * (x_coords[0] - x_coords[1])) / (cosines[1] * (tangents[0] - tangents[1]) + 10**-8)
        distances = np.where(tangents[0] == tangents[1], 0, distances)
        print(distances)

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
                if self.crystals[i].growing_left:
                    self.crystals[i].length_left += self.crystals[i].speed
        
        #self.root.after(200, self.render_crystals)

    def render_centers(self):
        for crystal in self.crystals:
            self.canvas.create_oval(crystal.center.x - 5, crystal.center.y - 5, crystal.center.x + 5, crystal.center.y + 5, fill="red")

    def create_window(self):
        
        self.render_crystals()
        self.render_centers()

        #self.root.after(200, self.render_crystals)

        self.root.mainloop()
        

    
master = WPattern()
master.calc_intersection_distances()
master.create_window()
