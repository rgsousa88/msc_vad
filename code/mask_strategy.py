import torch

import numpy as np
import os
import math

class MaskGeneratorTorch:
    def __init__(self, 
                 height: int, 
                 width: int, 
                 percent: float = None, 
                 num_squares: int = None, 
                 square_size: int = 10,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.height = height
        self.width = width
        self.device = device
        
        # Move computations to device
        self.height_tensor = torch.tensor(height, device=device)
        self.width_tensor = torch.tensor(width, device=device)
        
        self.area = height * width
        self.square_size = square_size
        self.square_area = square_size * square_size
        self.current = 0

        # Parameter validation and computation
        if percent is None and num_squares is None:
            self.percent = 0.30
            self._compute_squares()
        elif percent is None and num_squares is not None:
            self.num_squares = num_squares
            self._compute_percent()
        elif percent is not None and num_squares is None:
            self.percent = percent
            self._compute_squares()
        else:
            self.percent = percent
            self.num_squares = num_squares
            self._compute_square_size()
        
        self.radius = square_size // 2
        
        # Precompute valid ranges
        self.r_range = torch.arange(self.radius, self.height - self.radius, device=self.device)
        self.w_range = torch.arange(self.radius, self.width - self.radius, device=self.device)

        self.grid_r, self.grid_w = torch.meshgrid(self.r_range, self.w_range, indexing='ij')
        
        self.__reset_indices__()

    def _compute_squares(self) -> None:
        self.covered_area = self.percent * self.area
        self.num_squares = int(self.covered_area / self.square_area)
    
    def _compute_percent(self) -> None:
        self.covered_area = self.num_squares * self.square_area
        self.percent = self.covered_area / self.area
    
    def _compute_square_size(self) -> None:
        self.covered_area = self.percent * self.area
        self.square_area = self.covered_area / self.num_squares
        self.square_size = math.floor(math.sqrt(self.square_area))
        self.radius = self.square_size // 2

    def __reset_indices__(self,):
        self.indices = self.grid_r * self.width + self.grid_w
    
    def get_squares_coords(self,):
        """
        Creates r: List(int), c: List(int) coords for one single frame 
        """
        remaining_squares = self.num_squares
        pixel_coords_x = []
        pixel_coords_y = []

        while remaining_squares > 0:
            r = torch.randint(self.radius, self.height - self.radius, size=(1,))
            c = torch.randint(self.radius, self.width - self.radius, size=(1,))

            center_r = r-self.radius
            center_c = c-self.radius

            if self.indices[center_r,center_c] > 0:
                pixel_coords_x.append(r.cpu().item())
                pixel_coords_y.append(c.cpu().item())

                r_start = max(0, center_r - self.radius)
                r_end = min(self.height, center_r + self.radius + (1 if self.square_size % 2 == 0 else 0))
                c_start = max(0, center_c - self.radius)
                c_end = min(self.width, center_c + self.radius + (1 if self.square_size % 2 == 0 else 0))

                self.indices[r_start:r_end, c_start:c_end] = -1
                remaining_squares-=1
        
        self.__reset_indices__()
        return pixel_coords_x, pixel_coords_y

    def get_batched_squares_coords(self, n_batch=1, n_frame_per_batch=2):        
        batch_coords_x = []
        batch_coords_y = []

        for i in range(n_batch):
            for j in range(n_frame_per_batch):
                frame_coords_x, frame_coords_y = self.get_squares_coords()
                batch_coords_x.append(frame_coords_x)
                batch_coords_y.append(frame_coords_y)
        
        return batch_coords_x, batch_coords_y


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
    
    def get_batched_squares_coords(self, n_batch=1):        
        pixel_coords_x = []
        pixel_coords_y = []

        for i in range(n_batch):
            pixel_coord_x_batch = []
            pixel_coord_y_batch = []
            
            if len(self.map_index.keys()) == 0:
                self.__initialize_map__()
            
            for pixel in next(self):
                coord = np.unravel_index(pixel, (self.height, self.width))
                pixel_coord_x_batch.append(coord[0])
                pixel_coord_y_batch.append(coord[1])

            pixel_coords_x.extend(pixel_coord_x_batch)
            pixel_coords_y.extend(pixel_coord_y_batch)
            self.__reset__()
        
        return pixel_coords_x, pixel_coords_y