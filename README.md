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

vis_save的各种元素的可视化都是返回一个list，这些list相加起来可以组合出复杂的场景，最后保存的是由每一时刻的list组成的list，现在保存会得到一个文件夹，这个文件夹里面会有scene.json和一堆mesh，同样的mesh只会保存一次

vis_save可以在远程服务器上运行，在文件夹保存后可以在本地运行vis_pyvista，通过环境变量设置完远程服务器的路径（详见src/utils/download.py）后就会自动下载远程服务器的内容，之后可以使用鼠标自由调整视角，并且可以滑动右上角的进度条来切换显示的时间步（为了减小运算量，不同时间步同样的mesh和robot可以在vis_save的保存过程中设置同样的name）

此外还有一些键盘快捷键，包括

（数字）r：重新下载并读取
（数字）o：重新读取
（数字）d：重新下载
（数字）s：保存当前scene到其他目录
j，k：分别为下一帧与前一帧
空格：开始/暂停播放
z：重置时间为0
q：清除数字buffer

其中（数字）代表可以在输入对应指令前输入场景编号，这个数字可以指定下载/读取/保存的位置，不设定数字则默认为当前的场景编号（初始化为0）
