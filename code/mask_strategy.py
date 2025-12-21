import numpy as np
import os
import math

class MaskGenerator:
    def __init__(self, height: int, width: int, percent:float = None, num_squares:int = None, square_size:int=10):
        self.height = height
        self.width = width
        self.area = height * width
        self.square_size = square_size
        self.square_area = self.square_size * self.square_size
        self.current = 0

        if percent == None and num_squares == None:
            self.percent = 0.30
            self.__compute_squares__()
        elif percent == None and num_squares:
            self.num_squares = num_squares
            self.__compute_percent__()
        elif percent and num_squares == None:
            self.percent = percent
            self.__compute_squares__()
        else:
            self.percent = percent
            self.num_squares = num_squares
            self.__compute_square_size__()
        
        self.radius = int(self.square_size // 2)
        self.r_range = range(self.radius, self.height - self.radius)
        self.w_range = range(self.radius, self.width - self.radius)

        self.coords = []
        self.map_index = {}
        
        self.__initialize_map__()
    
    def __iter__(self,):
        return self

    def __compute_squares__(self,):
        self.covered_area = self.percent * self.area
        self.num_squares = int(self.covered_area / self.square_area)
    
    def __compute_percent__(self,):
        self.covered_area = self.num_squares * self.square_area
        self.percent = self.covered_area / self.area
    
    def __compute_square_size__(self,):
        self.covered_area = self.percent * self.area
        self.square_area = self.covered_area / self.num_squares
        self.square_size = math.floor(math.sqrt(self.square_area))

    def __str__(self,):
        msg = f"Heigth {self.height} Width {self.width} Area {self.area} Mask Type {self.mask_type}"
        msg = f"{msg}\nSquare Size {self.square_size} Square Area {self.square_area} Radius {self.radius}"
        msg = f"{msg}\nPercent {self.percent} Num Squares {self.num_squares} Covered Area {self.covered_area}"

        return msg
    
    def __create_neighboor__(self, i, j):
        indices = []
        for k in range(i-2*self.radius,i+2*self.radius):
            for m in range(j-2*self.radius,j+2*self.radius):
                indices.append(k*self.width + m)
        return indices

    def __reset__(self,):
        #print(f"Mask Generator Reset")
        self.current = 0
        self.coords.clear()
        self.map_index.clear()

    def __initialize_map__(self,):
        #print(f"Mask Generator Init")

        for i in self.r_range:
            for j in self.w_range:
                self.coords.append(f"Coord ({i},{j})")
                key = i*self.width + j
                if not key in self.map_index.keys():
                    self.map_index[key] = self.__create_neighboor__(i,j)

    def __next__(self,):
        if self.current > self.num_squares:
            self.current = 0
            raise StopIteration()
        
        if len(self.map_index.keys()) <= 0:
            raise StopIteration()

        selected_key = np.random.choice(list(self.map_index.keys()))
        square_indexes = self.map_index[selected_key]
        del self.map_index[selected_key]
        
        for index in square_indexes:
            if index in self.map_index.keys():
                del self.map_index[index]

        self.current+=1
        return selected_key
    
    def get_squares_coords(self,):
        #print(f"Mask Generator Get Coords")
        if len(self.map_index.keys()) == 0:
            self.__initialize_map__()
        
        pixel_coords_x = []
        pixel_coords_y = []

        for i, pixel in enumerate(self):
            coord = np.unravel_index(pixel, (self.height, self.width))
            pixel_coords_x.append(coord[0])
            pixel_coords_y.append(coord[1])

        self.__reset__()
        return pixel_coords_x, pixel_coords_y