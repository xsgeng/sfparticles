## 安装方法

### 克隆仓库
```bash
git clone http://172.16.95.131/gengxs/sfparticles.git
```

然后用户安装
```bash
cd sfparticles
# editable install
pip install --user -e .
```

### 直接安装
```bash
pip install --user git+http://172.16.95.131/gengxs/sfparticles.git
```

## 用法

级联过程的模拟参考`examples/cascade.py`

计算粒子轨迹的模拟参考`examples/trajectory.py`

设置环境变量`SFPARTICLES_OPTICAL_DEPTH=1`来使用光学深度辐射模型。
```bash
SFPARTICLES_OPTICAL_DEPTH=1 NUMBA_NUM_THREADS=20 python example/cascade.py
```
### GPU加速
设置环境变量`SFPARTICLES_USE_GPU=1`。
```bash
SFPARTICLES_USE_GPU=1 python example/cascade.py
# slurm
SFPARTICLES_USE_GPU=1 srun -p gpu -G 1 -u python example/cascade.py
```