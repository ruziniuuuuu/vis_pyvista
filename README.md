# vis_pyvista

## Installation

```bash
conda create -n vis python=3.9
conda activate vis
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia

wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch3d/linux-64/pytorch3d-0.7.8-py39_cu118_pyt241.tar.bz2
conda install -y --use-local ./pytorch3d-0.7.8-py39_cu118_pyt241.tar.bz2

pip install plotly transforms3d open3d==0.17.0 urdf_parser_py tensorboard wandb coacd rich
pip install numpy==1.26.4 pin pyvista trimesh==4.5.2 pycollada
```

## Usage

Example Usage:

```bash
# server
python src/utils/vis_save.py # the scene construction can be found in self_test
# local
IP=<IP address> PORT=<port number> REMOTE_USER=<remote server's username> KEY=<path to ssh key> REMOTE_PATH=<directory vis's path on remote server> LOCAL_PATH=./tmp/vis python src/utils/vis_pyvista.py
```

The visualization of various elements in vis_save returns a list. These lists can be combined to create complex scenes, and the final saved result is a list of lists, each representing a moment in time. Currently, saving will result in a folder containing scene.json and a bunch of meshes, with the same mesh being saved only once.

vis_save can run on a remote server. After saving the folder, you can run vis_pyvista locally. By setting the remote server's path through environment variables (see src/utils/download.py), the content from the remote server will be automatically downloaded. You can then freely adjust the view with the mouse and use the progress bar in the upper right corner to switch between time steps. 

Additionally, there are some keyboard shortcuts, including:

- (number) r: re-download and read
- (number) o: re-read
- (number) d: re-download
- (number) s: save the current scene to another directory
- j, k: next frame and previous frame, respectively
- space: start/pause playback
- z: reset time to 0
- q: clear the number buffer

The (number) represents the scene number that can be input before the corresponding command. This number can specify the download/read/save location. If no number is set, the current scene number is used (initialized to 0).
