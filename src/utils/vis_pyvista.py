import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.realpath('.'))

from typing import Optional, Union, Dict, List
import trimesh as tm
from tqdm import tqdm
import numpy as np
import torch
import pyvista as pv
import random
import shutil
import json
from time import sleep
from transforms3d.quaternions import quat2mat, mat2quat
import plotly.express as px

from src.utils.utils import to_numpy, to_torch, to_number, rm_r, safe_copy, serialize_item
from src.utils.pin_model import PinRobotModel
from src.utils.download import download_with_rsync

class Vis:
    def __init__(self, path):
        self.robots = dict()
        self.plotter = pv.Plotter()
        self.actors = dict()
        self.elements = dict()
        self.text_buffer = ''
        self.t = 0
        self.slider = None
        self.running = False
        self.path = path
        self.cur_scene = '0'

    def sphere(self,
               trans: Optional[Union[np.ndarray, torch.tensor]] = None,
               radius: float = None,
               opacity: float = None,
               color: str = None,
               name: str = None,
               idx: int = None,
    ):
        color = 'blue' if color is None else color
        opacity = 1.0 if opacity is None else opacity
        trans = np.zeros(3) if trans is None else to_numpy(trans)
        radius = 0.1 if radius is None else to_number(radius)

        sphere = pv.Sphere(radius=radius, center=trans)
        self.elements[f'{name}_t{idx}'] = dict(param={'mesh': sphere, 'color': color, 'opacity': opacity}, idx=[idx], mesh_pose=None)
        return
    
    @staticmethod
    def rand_color():
        return random.choice(px.colors.sequential.Plasma)
    
    def box(self,
                   scale: Union[np.ndarray, torch.tensor], # (3, )
                   trans: Optional[Union[np.ndarray, torch.tensor]] = None, # (3, )
                   rot: Optional[Union[np.ndarray, torch.tensor]] = None, # (3, 3)
                   opacity: Optional[float] = None,
                   color: Optional[str] = None,
                   name: str = None,
                   idx: int = None,
        ):

        color = 'violet' if color is None else color
        opacity = 1.0 if opacity is None else opacity
        trans = np.zeros(3) if trans is None else to_numpy(trans)
        rot = np.eye(3) if rot is None else to_numpy(rot)

        box = pv.Cube(center=np.zeros(3), x_length=scale[0], y_length=scale[1], z_length=scale[2])
        box.transform(np.block([[rot, trans[:, None]], [np.zeros(3), 1]]))
        self.elements[f'{name}_t{idx}'] = dict(param={'mesh': box, 'color': color, 'opacity': opacity}, idx=[idx], mesh_pose=None)
        return
    
    def pose(self,  
                    trans: Union[np.ndarray, torch.tensor], # (3, )
                    rot: Union[np.ndarray, torch.tensor], # (3, 3)
                    width: int = 0.1,
                    length: float = 0.01,
                    name: str = None,
                    idx: int = None,
    ):
        trans = to_numpy(trans)
        rot = to_numpy(rot)
        arrow_x = pv.Arrow(start=trans, direction=rot[:, 0], tip_length=length, scale=width)
        arrow_y = pv.Arrow(start=trans, direction=rot[:, 1], tip_length=length, scale=width)
        arrow_z = pv.Arrow(start=trans, direction=rot[:, 2], tip_length=length, scale=width)
        self.elements[f'{name}_t{idx}x'] = dict(param={'mesh': arrow_x, 'color': 'red', 'opacity': 1.0}, idx=[idx], mesh_pose=None)
        self.elements[f'{name}_t{idx}y'] = dict(param={'mesh': arrow_y, 'color': 'green', 'opacity': 1.0}, idx=[idx], mesh_pose=None)
        self.elements[f'{name}_t{idx}z'] = dict(param={'mesh': arrow_z, 'color': 'blue', 'opacity': 1.0}, idx=[idx], mesh_pose=None)
        return
   
    def plane(self,
                     plane_vec: Union[np.ndarray, torch.tensor], # (4, )
                     opacity: Optional[float] = None,
                     color: Optional[str] = None,
                     name: str = None,
                     idx: int = None,
    ):
        plane_vec = to_torch(plane_vec)
        color = 'blue' if color is None else color
        opacity = 1.0 if opacity is None else opacity
        dir = plane_vec[:3]
        assert (torch.linalg.norm(dir) - 1).abs() < 1e-4
        plane = pv.Plane(direction=plane_vec[:3], center=plane_vec[:3] * plane_vec[3])
        self.elements[f'{name}_t{idx}'] = dict(param={'mesh': plane, 'color': color, 'opacity': opacity}, idx=[idx], mesh_pose=None)

    def robot(self,
              urdf: str,
              qpos: Union[Union[np.ndarray, torch.tensor]], # (n)
              trans: Optional[Union[np.ndarray, torch.tensor]] = None, # (3)
              rot: Optional[Union[np.ndarray, torch.tensor]] = None, # (3, 3)
              opacity: Optional[float] = None,
              color: Optional[str] = None,
              mesh_type: str = 'collision',
              name: str = None,
              idx: int = None,
    ):
        trans = np.zeros((3,)) if trans is None else to_numpy(trans).reshape(3)
        rot = np.eye(3) if rot is None else to_numpy(rot).reshape(3, 3)
        qpos = to_numpy(qpos).reshape(-1)
        color = 'violet' if color is None else color
        opacity = 1.0 if opacity is None else opacity
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = trans

        if urdf not in self.robots:
            self.robots[urdf] = PinRobotModel(urdf)
        poses = self.robots[urdf].forward_kinematics(qpos, mesh_type)
        for mesh_id, (mesh, (mesh_trans, mesh_rot)) in enumerate(zip(self.robots[urdf].meshes[mesh_type], poses)):
            vertices, faces = mesh.vertices, mesh.faces
            self.mesh(vertices=vertices, faces=faces, trans=rot@mesh_trans+trans, rot=rot@mesh_rot, opacity=opacity, color=color, name=f'{name}_mesh_id{mesh_id}', idx=idx)

    def pc(self,
                  pc: Union[np.ndarray, torch.tensor], # (n, 3)
                  value: Optional[Union[np.ndarray, torch.tensor]] = None, # (n, )
                  size: int = 1,
                  color: Union[str, Union[np.ndarray, torch.tensor]] = 'red', # (n, 3)
                  color_map: str = 'Viridis',
                  name: str = None,
                  idx: int = None,
    ):
        pc = to_numpy(pc)
        cloud = pv.PolyData(pc)
        pc_name = f'{name}_t{idx}'
        self.elements[pc_name] = dict(param={'mesh': cloud}, idx=[idx], mesh_pose=None)
        if not value is None:
            self.elements[pc_name]['param']['scalars'] = to_numpy(value)
            self.elements[pc_name]['param']['show_scalar_bar'] = False
        if not isinstance(color, str):
            color = to_numpy(color)
            if color.shape[-1] == 3:
                color = np.concatenate([color, np.ones((len(color), 1))], axis=-1)
            self.elements[pc_name]['param']['scalars'] = color
            self.elements[pc_name]['param']['rgba'] = True
        else:
            self.elements[pc_name]['param']['color'] = color
            self.elements[pc_name]['param']['opacity'] = 1.0
        return
    
    def line(self,
                    p1: Union[np.ndarray, torch.tensor], # (3)
                    p2: Union[np.ndarray, torch.tensor], # (3)
                    width: int = None,
                    color: str = None,
                    name: str = None,
                    idx: int = None,
    ):
        color = 'green' if color is None else color
        width = 1 if width is None else width

        p1, p2 = to_numpy(p1), to_numpy(p2)
        line = pv.Line(pointa=p1, pointb=p2)
        self.elements[f'{name}_t{idx}'] = dict(param={'mesh': line, 'color': color, 'line_width': width}, idx=[idx], mesh_pose=None)
    
    def mesh(self,
                    path: str = None,
                    scale: float = 1.0,
                    trans: Optional[Union[np.ndarray, torch.tensor]] = None, # (3, )
                    rot: Optional[Union[np.ndarray, torch.tensor]] = None, # (3, 3)
                    opacity: float = 1.0,
                    color: str = 'lightgreen',
                    vertices: Optional[Union[np.ndarray, torch.tensor]] = None, # (n, 3)
                    faces: Optional[Union[np.ndarray, torch.tensor]] = None, # (m, 3)
                    name: str = None,
                    idx: int = None,
    ):
        trans = np.zeros(3) if trans is None else to_numpy(trans)
        rot = np.eye(3) if rot is None else to_numpy(rot)
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = trans
        elem_name = f'{name}_t'
        if elem_name in self.elements:
            self.elements[elem_name]['idx'].append(idx)
            self.elements[elem_name]['mesh_pose'][idx] = pose
        else:
            if path is not None:
                mesh = tm.load_mesh(path)
                vertices, faces = mesh.vertices, mesh.faces
            vertices = to_numpy(vertices)
            faces = to_numpy(faces)
            faces = [[len(f)] + f for f in faces.tolist()]
            mesh = pv.PolyData(vertices, faces)
            self.elements[elem_name] = dict(param={'mesh': mesh, 'color': color, 'opacity': opacity}, idx=[idx], mesh_pose={idx: pose})
    
    def read(self, path: str):
        scenes = json.load(open(os.path.join(path, 'scene.json'), 'r'))
        self.T = len(scenes)
        for idx, scene in enumerate(scenes):
            for o in scene:
                if o['type'] == 'sphere':
                    self.sphere(trans=o['trans'], radius=o['radius'], opacity=o['opacity'], color=o['color'], name=o['name'], idx=idx)
                if o['type'] == 'box':
                    self.box(scale=o['size'], trans=o['pos'], rot=quat2mat(o['quat']), opacity=o['opacity'], color=o['color'], name=o['name'], idx=idx)
                if o['type'] == 'pose':
                    self.pose(trans=o['trans'], rot=o['rot'], width=o['width'], length=o['length'], name=o['name'], idx=idx)
                if o['type'] == 'plane':
                    self.plane(plane_vec=o['plane'], opacity=o['opacity'], color=o['color'], name=o['name'], idx=idx)
                if o['type'] == 'robot':
                    self.robot(urdf=o['urdf'], qpos=o['qpos'], trans=o['pos'], rot=quat2mat(o['quat']), opacity=o['opacity'], color=o['color'], mesh_type=o['mesh_type'], name=o['name'], idx=idx)
                if o['type'] == 'pc':
                    self.pc(pc=o['pc'], value=o['value'], size=o['size'], color=o['color'], color_map=o['color_map'], name=o['name'], idx=idx)
                if o['type'] == 'line':
                    self.line(p1=o['p1'], p2=o['p2'], width=o['width'], color=o['color'], name=o['name'], idx=idx)
                if o['type'] == 'mesh':
                    self.mesh(path=os.path.join(path, o['path']), scale=o['scale'], trans=o['pos'], rot=quat2mat(o['quat']), opacity=o['opacity'], color=o['color'], vertices=o['vertices'], faces=o['faces'], name=o['name'], idx=idx)
        return

    def show(self):
        if self.last_t != self.t:
            for k, v in self.elements.items():
                if self.t in v['idx']:
                    if not k in self.actors:
                        self.actors[k] = self.plotter.add_mesh(**v['param'])
                    else:
                        self.actors[k].visibility = True
                    if not v['mesh_pose'] is None:
                        self.actors[k].user_matrix = v['mesh_pose'][self.t]
                elif k in self.actors:
                    self.actors[k].visibility = False
        self.last_t = self.t
        if not self.scene_has_shown:
            self.slider = self.plotter.add_slider_widget(
                self.update_visualization, 
                rng=[0, self.T - 1], 
                value=self.t, 
                title="", 
                style="modern",
                pointa=(0.65, 0.95),
                pointb=(0.95, 0.95),
                slider_width=0.02,
                tube_width=0.02,
                interaction_event="always",
            )
            self.scene_has_shown = True
        if not self.has_shown:
            self.plotter.add_key_event('r', self.reload_elements)
            self.plotter.add_key_event('o', self.load_elements)
            self.plotter.add_key_event('s', self.save_elements)
            self.plotter.add_key_event('d', self.download_elements)
            self.plotter.add_key_event('j', self.next_elements)
            self.plotter.add_key_event('k', self.last_elements)
            self.plotter.add_key_event('space', self.animate)
            self.plotter.add_key_event('z', self.reset_t)
            self.plotter.add_key_event('q', self.reset_buffer)
            for i in range(10):  # Loop through '0' to '9'
                key = str(i)  # Convert the number to string (e.g., '0', '1', ..., '9')
                self.plotter.add_key_event(key, lambda c=key: self.add_buffer(c))
            self.plotter.enable_terrain_style()
            self.has_shown = True
            self.plotter.show()
        else:
            self.plotter.update()
    
    def add_buffer(self, c):
        self.text_buffer += c
    
    def reset_buffer(self):
        self.text_buffer = ''
    
    def reset_t(self):
        self.update_visualization(0)
    
    def animate(self):
        if self.running:
            self.running = False
        else:
            self.running = True
            for t in range(self.t, self.T):
                if not self.running:
                    break
                self.update_visualization(t)
                sleep(0.05)

    def update_visualization(self, t):
        if self.has_shown:
            self.t = int(t)
            self.slider.GetRepresentation().SetValue(self.t)
            self.show()

    def last_elements(self):
        self.update_visualization(max(self.t - 1, 0))

    def next_elements(self):
        self.update_visualization(min(self.t + 1, self.T - 1))
    
    def get_scene_id_path(self, scene_id=None):
        if scene_id is None:
            scene_id = self.text_buffer
            if scene_id == '':
                scene_id = self.cur_scene
            self.text_buffer = ''
        scene_path = self.path + "_" + scene_id
        return scene_id, scene_path
        
    def download_elements(self, scene_path=None):
        if scene_path is None:
            _, scene_path = self.get_scene_id_path()
        download_with_rsync(local_dir=scene_path)

    def load_elements(self, scene_id=None, scene_path=None):
        if scene_path is None:
            scene_id, scene_path = self.get_scene_id_path()
        self.cur_scene = scene_id
        self.close_last()
        self.read(scene_path)
        self.scene_has_shown = False
        self.show()
    
    def reload_elements(self):
        scene_id, scene_path = self.get_scene_id_path()
        self.download_elements(scene_path=scene_path)
        self.load_elements(scene_id=scene_id, scene_path=scene_path)

    def save_elements(self):
        _, old_scene_path = self.get_scene_id_path(scene_id=self.cur_scene)
        _, new_scene_path = self.get_scene_id_path()
        safe_copy(old_scene_path, new_scene_path)

    def close_last(self):
        self.reset_buffer()
        self.t = 0
        self.last_t = -1
        for v in self.actors.values():
            self.plotter.remove_actor(v)
        self.elements = dict()
        self.actors = dict()
        self.running = False
        if not self.slider is None:
            self.plotter.clear_slider_widgets()
            self.slider = None
 
    def read_show(self):
        self.has_shown = False
        self.reload_elements()


if __name__ == '__main__':
    vis = Vis('tmp/vis')
    vis.read_show()