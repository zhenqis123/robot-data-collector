import sys
# sys.path.append(r"/home/zbh/MANUS_Core_3.0.1_SDK/ros2_ws/install/manus_ros2_msgs/local/lib/python3.10/dist-packages")
import threading
import numpy as np
import rclpy
from loop_rate_limiters import RateLimiter
from manus_ros2_msgs.msg import ManusGlove
from rclpy.node import Node
import socket
import struct
import time


class MinimalSubscriber(Node):
    def __init__(self, glove_indexes):
        super().__init__("manus_ros2_client_py")

        # Define glove ID to side mapping
        self.glove_side_map = {-1624338395: "left", 990853568: "right"} # 0 for Left, 1 for Right
        self.latest_positions = {}

        # Subscribe to glove data topics for each glove_id
        self.sub_poses = []
        for glove_id in glove_indexes:
            topic_name = f"/manus_glove_{glove_id}"
            self.sub_poses.append(self.create_subscription(
                ManusGlove,
                topic_name,
                self.glove_callback,
                20,
            ))

        # UDP Config
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_target = ("127.0.0.1", 6667)

    def glove_callback(self, msg: ManusGlove):
        """Callback for glove node poses"""
        positions = []
        orientations = []

        # Update node visualization
        for node in msg.raw_nodes:
            pose = node.pose
            
            positions.append([-pose.position.x, pose.position.y, pose.position.z])
            orientations.append([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])

        # Store latest positions for this glove
        side = self.glove_side_map.get(msg.glove_id, f"unknown_{msg.glove_id}")
        self.latest_positions[f"{side}_fingers"] = np.array(positions)
        self.latest_positions[f"{side}_orientations"] = np.array(orientations)

        # Publish merged data from all gloves
        self.send_udp_packet()

    def send_udp_packet(self):
        # 1. Timestamp (double)
        packet = struct.pack('<d', time.time())
        
        # Helper for packing arrays
        def pack_array(key, shape):
            arr = self.latest_positions.get(key)
            if arr is None or arr.shape != shape:
                return np.zeros(shape, dtype=np.float32).tobytes()
            return arr.astype(np.float32).tobytes()

        # 2. Left Fingers (25*3)
        packet += pack_array('left_fingers', (25, 3))
        # 3. Left Orientations (25*4)
        packet += pack_array('left_orientations', (25, 4))
        # 4. Right Fingers (25*3)
        packet += pack_array('right_fingers', (25, 3))
        # 5. Right Orientations (25*4)
        packet += pack_array('right_orientations', (25, 4))
        
        try:
            self.udp_sock.sendto(packet, self.udp_target)
        except Exception:
            pass



def spin_node(glove_indexes):
    rclpy.init(args=sys.argv)

    minimal_subscriber = MinimalSubscriber(glove_indexes)
    rclpy.spin(minimal_subscriber)

    minimal_subscriber.destroy_node()
    rclpy.shutdown()


def main():
    glove_indexes = [0, 1]  # Add more glove indexes as needed

    spin_thread = threading.Thread(target=spin_node, args=(glove_indexes,), daemon=True)
    spin_thread.start()

    rate = RateLimiter(frequency=120.0, warn=False)
    while True:
        rate.sleep()


if __name__ == "__main__":
    main()
