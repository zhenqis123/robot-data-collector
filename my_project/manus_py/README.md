# Enable dexterous tool using and manipulation

## Environment Setup
- Install IsaacGym preview 4.0 (https://developer.nvidia.com/isaac-gym).
- Manus Glove: 
    - Install ROS2 and sdk (https://docs.manus-meta.com/3.0.0/Plugins/SDK/ROS2/getting%20started/)
    - Install dex-retargeting: download dex-retargeting.zip from (https://disk.pku.edu.cn/link/AAE98116E3105443D3BC690B3D1B32CF3B), unzip and `pip install -e .`
    - `pip install sapien==3.0.0b0`


## Manipulation skills and skill-specific teleoperation

- Test the IsaacGym environment: `python run_test_env.py num_envs=1 +debug=check_joint`

- Test the Manus glove:
    - After install Manus ROS2 sdk, move `manus/manus_data_viz.py` to your `.../ros2_ws/src/manus_ros2/client_scripts/`, move `manus/run_manus_all.zsh` to your `.../ros2_ws/`.
    Connect the glove in either wired or wireless mode.
    `cd $YOUR_PATH/ros2_ws`, run `./run_manus_all.zsh`.
    - `cd manus`, run `python manus_dex_retarget.py  --mode=right --robot_name wuji --render_sapien --retargeting_type vector`. You can see the retargeted wuji hand in the sapien simulator.

- After you get familiar with these code, TODO:
    - Add a redis listener in run_test_env.py to teleoperate the hand in IsaacGym using Manus glove.
    - Modify the initial wrist pose and object pose in IsaacGym to enable in-hand manipulation.


