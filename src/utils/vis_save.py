import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.realpath('.'))

from typing import Optional, Union, Dict, List
import trimesh as tm
from tqdm import tqdm
import numpy as np
import torch
import plotly.express as px
import random
import shutil
import json
from transforms3d.quaternions import quat2mat, mat2quat

from src.utils.utils import to_numpy, to_torch, to_number, rm_r, safe_copy, serialize_item, gen_uuid

class Vis:
    def __init__(self, *args, **kwargs):
        pass
    
    @staticmethod
    def save(lst: Union[List[Dict], List[List[Dict]]], dir: str = 'tmp/vis'):
        if isinstance(lst[0], dict):
            lst = [lst]
        # empty directory
        rm_r(dir)
        os.makedirs(dir, exist_ok=True)
        for lt in lst:
            for o in lt:
                if o['type'] == 'mesh':
                    if not o['path'] is None:
                        safe_copy(o['path'], os.path.join(dir, o['path']))
                    if not o['urdf'] is None:
                        urdf_dir = os.path.dirname(o['urdf'])
                        safe_copy(o['urdf'], os.path.join(dir, o['urdf']))
        with open(os.path.join(dir, 'scene.json'), 'w') as f:
            json.dump(serialize_item(lst), f, indent=4)
        print(f'Saved to {dir}')

    @staticmethod
    def sphere(trans: Optional[Union[np.ndarray, torch.tensor]] = None,
               radius: float = None,
               opacity: float = None,
               color: str = None,
               name: str = None,
    ) -> list:
        color = 'blue' if color is None else color
        opacity = 1.0 if opacity is None else opacity
        trans = np.zeros(3) if trans is None else to_numpy(trans)
        radius = 0.1 if radius is None else to_number(radius)
        name = gen_uuid() if name is None else name

        return [dict(type='sphere',
                     color=color,
                     opacity=opacity,
                     radius=radius,
                     trans=trans,
                     name=name)]
    
    @staticmethod
    def rand_color():
        return random.choice(px.colors.sequential.Plasma)
    
    @staticmethod
    def box(scale: Union[np.ndarray, torch.tensor], # (3, )
                   trans: Optional[Union[np.ndarray, torch.tensor]] = None, # (3, )
                   rot: Optional[Union[np.ndarray, torch.tensor]] = None, # (3, 3)
                   opacity: Optional[float] = None,
                   color: Optional[str] = None,
                   name: str = None,
        ) -> list:

        color = 'violet' if color is None else color
        opacity = 1.0 if opacity is None else opacity
        trans = np.zeros(3) if trans is None else to_numpy(trans)
        rot = np.eye(3) if rot is None else to_numpy(rot)
        name = gen_uuid() if name is None else name

        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = trans

        return [dict(type='box',
                     color=color,
                     opacity=opacity,
                     size=scale,
                     pos=trans,
                     quat=mat2quat(rot),
                     pose=pose,
                     name=name)]
    
    @staticmethod
    def pose(trans: Union[np.ndarray, torch.tensor], # (3, )
                    rot: Union[np.ndarray, torch.tensor], # (3, 3)
                    width: int = 0.1,
                    length: float = 0.01,
                    name: str = None,
    ) -> list:
        name = gen_uuid() if name is None else name
        return [dict(type='pose',
                     trans=trans,
                     rot=rot,
                     width=width,
                     length=length,
                     name=name)]
    
    @staticmethod
    def plane(plane_vec: Union[np.ndarray, torch.tensor], # (4, )
                     opacity: Optional[float] = None,
                     color: Optional[str] = None,
                     name: str = None,
    ) -> list:
        plane_vec = to_torch(plane_vec)
        color = 'blue' if color is None else color
        opacity = 1.0 if opacity is None else opacity
        dir = plane_vec[:3]
        name = gen_uuid() if name is None else name
        assert (torch.linalg.norm(dir) - 1).abs() < 1e-4
        return [dict(type='plane',
                    color=color,
                    opacity=opacity,
                    plane=plane_vec,
                    name=name)]

    @staticmethod
    def robot(urdf: str,
              qpos: Union[Union[np.ndarray, torch.tensor]], # (n)
              trans: Optional[Union[np.ndarray, torch.tensor]] = None, # (3)
              rot: Optional[Union[np.ndarray, torch.tensor]] = None, # (3, 3)
              opacity: Optional[float] = None,
              color: Optional[str] = None,
              mesh_type: str = 'collision',
              name: str = None,
    ) -> list:
        trans = np.zeros((3,)) if trans is None else to_numpy(trans).reshape(3)
        rot = np.eye(3) if rot is None else to_numpy(rot).reshape(3, 3)
        qpos = to_numpy(qpos).reshape(-1)
        color = 'violet' if color is None else color
        opacity = 1.0 if opacity is None else opacity
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = trans
        name = gen_uuid() if name is None else name

        return [dict(type='robot',
                     urdf=urdf,
                     color=color,
                     opacity=opacity,
                     qpos=qpos,
                     pos=trans,
                     quat=mat2quat(rot),
                     pose=pose,
                     mesh_type=mesh_type,
                     name=name)]
    
    @staticmethod
    def pc(pc: Union[np.ndarray, torch.tensor], # (n, 3)
                  value: Optional[Union[np.ndarray, torch.tensor]] = None, # (n, )
                  size: int = 1,
                  color: Union[str, Union[np.ndarray, torch.tensor]] = 'red', # (n, 3)
                  color_map: str = 'Viridis',
                  name: str = None,
    ) -> list:
        pc = to_numpy(pc)
        if not value is None:
            value = to_numpy(value)
        if not isinstance(color, str):
            color = to_numpy(color)
        name = gen_uuid() if name is None else name
        
        return [dict(type='pc',
                     pc=pc,
                     value=value,
                     size=size,
                     color=color,
                     color_map=color_map,
                     name=name)]
    
    @staticmethod
    def line(p1: Union[np.ndarray, torch.tensor], # (3)
                    p2: Union[np.ndarray, torch.tensor], # (3)
                    width: int = None,
                    color: str = None,
                    name: str = None,
    ) -> list:
        color = 'green' if color is None else color
        width = 1 if width is None else width
        name = gen_uuid() if name is None else name

        p1, p2 = to_numpy(p1), to_numpy(p2)
        return [dict(type='line',
                        p1=p1,
                        p2=p2,
                        width=width,
                        color=color,
                        name=name)]
    
    @staticmethod
    def mesh(path: str = None,
                    scale: float = 1.0,
                    trans: Optional[Union[np.ndarray, torch.tensor]] = None, # (3, )
                    rot: Optional[Union[np.ndarray, torch.tensor]] = None, # (3, 3)
                    opacity: float = 1.0,
                    color: str = 'lightgreen',
                    vertices: Optional[Union[np.ndarray, torch.tensor]] = None, # (n, 3)
                    faces: Optional[Union[np.ndarray, torch.tensor]] = None, # (m, 3)
                    name: str = None,
    ) -> list:
        trans = np.zeros(3) if trans is None else to_numpy(trans)
        rot = np.eye(3) if rot is None else to_numpy(rot)
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = trans
        name = gen_uuid() if name is None else name
        if path is None:
            vertices = to_numpy(vertices)
            faces = to_numpy(faces)
        
        return [dict(type='mesh',
                     path=path,
                     urdf=None,
                     scale=scale,
                     pos=trans,
                     quat=mat2quat(rot),
                     opacity=opacity,
                     color=color,
                     vertices=vertices,
                     faces=faces,
                     pose=pose,
                     name=name)]

    @staticmethod
    def self_test():
        from pytorch3d import transforms as pttf
        # each element in all_vis represents one timestep of the scene
        all_vis = []
        # one timestep can have multiple objects, we can add the lists of objects together
        all_vis.append(Vis.sphere(trans=np.random.randn(3)/10, opacity=0.9) + Vis.box(trans=np.random.randn(3)/10, rot=pttf.random_rotation(), scale=np.random.rand(3)/10+0.001, opacity=0.9))
        all_vis.append(Vis.box(trans=np.random.randn(3)/10, rot=pttf.random_rotation(), scale=np.random.rand(3)/10+0.001, opacity=0.9))
        all_vis.append(Vis.pose(trans=np.random.randn(3)/10, rot=pttf.random_rotation()))
        all_vis.append(Vis.plane(torch.cat([pttf.random_rotation()[0], torch.randn(1,)/10]), opacity=0.9))
        all_vis.append(Vis.pc(np.random.randn(100, 3)/10))
        all_vis.append(Vis.pc(np.random.randn(100, 3)/10, value=np.random.randn(100)))
        all_vis.append(Vis.pc(np.random.randn(100, 3)/10, color=np.random.rand(100, 3)))
        all_vis.append(Vis.pc(np.random.randn(100, 3)/10, color=np.random.rand(100, 4)))
        all_vis.append(Vis.line(np.random.randn(3)/10, np.random.randn(3)/10))
        Vis.save(all_vis, 'tmp/vis')

if __name__ == '__main__':
    Vis.self_test()