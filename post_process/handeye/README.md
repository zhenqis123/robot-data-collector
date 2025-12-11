## 眼在手外 手眼标定

### TODO
- 睿尔曼换了新sdk，代码相应的部分改一下（utils.ArmCalib）
- utils.CAMERA_IDS 改成实际D435相机的序列号

### 流程

把标定板固连在机械臂末端，机械臂开机能拖动即可。运行手眼标定：
```
python run_calibrate_handeye.py --camera 0 # 将相机的序列号放在utils.CAMERA_IDS[0]位置
```
跳出RGB窗口后，拖动机械臂末端使标定板在画面中被检测到（出现绿色检测框）。反复调整末端位姿（根据经验，在同个位置采不同旋转标出来效果更好），(将鼠标焦点置于RGB窗口)按r记录数据，重复10个点左右。最后按c计算存下变换。

可视化标定效果：
```
python vis_handeye.py --camera 0
```
同时渲染相机坐标系、base坐标系、base坐标系下的场景点云。
