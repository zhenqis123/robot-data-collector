import open3d as o3d
import pandas as pd
import numpy as np
import argparse
import os
import time
import sys

class ManusReplayer:
    def __init__(self, data_dir, fps=30.0):
        self.data_dir = data_dir
        self.csv_path = os.path.join(data_dir, "manus_data.csv")
        
        if not os.path.exists(self.csv_path):
            print(f"Error: {self.csv_path} not found.")
            sys.exit(1)
            
        print(f"Loading data from {self.csv_path}...")
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} frames.")
        
        self.target_fps = fps
        self.frame_interval = 1.0 / fps
        self.current_frame = 0
        self.is_paused = False
        self.num_joints = 25
        
        # Open3D Visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Manus Replay", width=1280, height=720)
        
        # Coordinate frame
        self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        self.vis.add_geometry(self.axis)
        
        # Initialize Hand Geometries
        self.lh_meshes = []
        self.rh_meshes = []
        self.lh_lines = None # TODO: Implement if topology is known
        self.rh_lines = None
        
        # Colors
        self.lh_color = [1.0, 0.5, 0.0] # Orange for Left
        self.rh_color = [0.0, 0.5, 1.0] # Blue for Right
        
        self._init_geometries()
        
        # Setup callbacks
        self.vis.register_key_callback(32, self._toggle_pause) # Space
        self.vis.register_key_callback(262, self._next_frame)  # Right Arrow
        self.vis.register_key_callback(263, self._prev_frame)  # Left Arrow
        
        # View control settings
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.8)
        
    def _init_geometries(self):
        # Create spheres for each joint
        for i in range(self.num_joints):
            # Left Hand
            mesh_l = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
            mesh_l.paint_uniform_color(self.lh_color)
            mesh_l.compute_vertex_normals()
            self.vis.add_geometry(mesh_l)
            self.lh_meshes.append(mesh_l)
            
            # Right Hand
            mesh_r = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
            mesh_r.paint_uniform_color(self.rh_color)
            mesh_r.compute_vertex_normals()
            self.vis.add_geometry(mesh_r)
            self.rh_meshes.append(mesh_r)
            
        # Lines (Optional, simplistic connection based on IDs if assumed sequential)
        # Without exact parent_id map from CSV, we skip lines to avoid mess.
            
    def _toggle_pause(self, vis):
        self.is_paused = not self.is_paused
        print(f"Paused: {self.is_paused}")
        return False
        
    def _next_frame(self, vis):
        if self.current_frame < len(self.df) - 1:
            self.current_frame += 1
            self._update_geometry()
        return False
        
    def _prev_frame(self, vis):
        if self.current_frame > 0:
            self.current_frame -= 1
            self._update_geometry()
        return False

    def _get_joint_pos(self, row, hand_prefix, joint_idx):
        # hand_prefix: 'lh' or 'rh'
        # column names: {prefix}_j{index}_p{axis}
        px = row[f'{hand_prefix}_j{joint_idx}_px']
        py = row[f'{hand_prefix}_j{joint_idx}_py']
        pz = row[f'{hand_prefix}_j{joint_idx}_pz']
        
        # Apply visualization transform (mirror X to match existing viz logic if needed)
        # In manus_data_viz.py: [-pose.position.x, pose.position.y, pose.position.z]
        return np.array([-px, py, pz])

    def _update_geometry(self):
        if self.current_frame >= len(self.df):
            self.current_frame = 0 # Loop
            
        row = self.df.iloc[self.current_frame]
        
        # Update Left Hand
        for i, mesh in enumerate(self.lh_meshes):
            pos = self._get_joint_pos(row, "lh", i)
            # Reset and translate
            mesh.translate(-mesh.get_center(), relative=True)
            mesh.translate(pos, relative=True)
            self.vis.update_geometry(mesh)
            
        # Update Right Hand
        for i, mesh in enumerate(self.rh_meshes):
            pos = self._get_joint_pos(row, "rh", i)
            mesh.translate(-mesh.get_center(), relative=True)
            mesh.translate(pos, relative=True)
            self.vis.update_geometry(mesh)
            
        self.vis.poll_events()
        self.vis.update_renderer()

    def run(self):
        print("Starting replay...")
        print("Controls: Space=Pause/Resume, Left/Right=Step Frames, ESC=Exit")
        
        try:
            while True:
                start_time = time.time()
                
                if not self.is_paused:
                    self.current_frame += 1
                    if self.current_frame >= len(self.df):
                        self.current_frame = 0 # Loop or stop
                        
                    self._update_geometry()
                
                # Frame limiting
                elapsed = time.time() - start_time
                wait = self.frame_interval - elapsed
                if wait > 0:
                    time.sleep(wait)
                    
                # Keep window responsive even when paused
                if self.is_paused:
                    self.vis.poll_events()
                    self.vis.update_renderer()
                    time.sleep(0.01)
                    
                if not self.vis.poll_events(): # Window closed
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description="Replay Manus Glove data from log folder")
    parser.add_argument("log_dir", default="/home/zc/HandMotion/temp_new/temp/robot-data-collector/my_project/resources/logs/captures/2026-01-14/sess_20260114_163528", help="Path to the capture session directory (containing manus_data.csv)")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS")
    
    args = parser.parse_args()
    
    replayer = ManusReplayer(args.log_dir, args.fps)
    replayer.run()

if __name__ == "__main__":
    main()
