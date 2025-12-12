# 可选配置
HAND = "DFQ"  # "Dex3-1"、"DFQ"、"FTP"

# 自动配置
if HAND == "Dex3-1":
    MODEL_NAME_LEFT = "dex3_1_l"
    MODEL_NAME_RIGHT = "dex3_1_r"
elif HAND == "DFQ":
    MODEL_NAME_LEFT = "DFQ_left_hand"
    MODEL_NAME_RIGHT = "DFQ_right_hand"
elif HAND == "FTP":
    MODEL_NAME_LEFT = "FTP_left_hand"
    MODEL_NAME_RIGHT = "FTP_right_hand"
else:
    print("配置有误！")

MODEL_PATH_LEFT = 'robots/%s.urdf' % (MODEL_NAME_LEFT)
MODEL_PATH_RIGHT = 'robots/%s.urdf' % (MODEL_NAME_RIGHT)

NAMES_JOINT_DEX31_LEFT = [
    ["left_hand_thumb_0_joint",],
    ["left_hand_thumb_1_joint",],
    ["left_hand_thumb_2_joint",],
    ["left_hand_index_0_joint",],
    ["left_hand_index_1_joint",],
    ["left_hand_middle_0_joint",],
    ["left_hand_middle_1_joint",],
]
NAMES_JOINT_DEX31_RIGHT = [
    ["right_hand_thumb_0_joint",],
    ["right_hand_thumb_1_joint",],
    ["right_hand_thumb_2_joint",],
    ["right_hand_index_0_joint",],
    ["right_hand_index_1_joint",],
    ["right_hand_middle_0_joint",],
    ["right_hand_middle_1_joint",],
]
NAMES_JOINT_INSPIREDHAND_LEFT = [
    # 0
    ["L_thumb_proximal_yaw_joint", "left_thumb_1_joint"],
    ["L_thumb_proximal_pitch_joint", "left_thumb_2_joint"],
    ["L_thumb_intermediate_joint", "left_thumb_3_joint"],
    ["L_thumb_distal_joint", "left_thumb_4_joint"],
    ["L_index_proximal_joint", "left_index_1_joint"],
    # 5
    ["L_index_intermediate_joint", "left_index_2_joint"],
    ["L_middle_proximal_joint", "left_middle_1_joint"],
    ["L_middle_intermediate_joint", "left_middle_2_joint"],
    ["L_ring_proximal_joint", "left_ring_1_joint"],
    ["L_ring_intermediate_joint", "left_ring_2_joint"],
    # 10
    ["L_pinky_proximal_joint", "left_little_1_joint"],
    ["L_pinky_intermediate_joint", "left_little_2_joint"]
]
NAMES_JOINT_INSPIREDHAND_RIGHT = [
    # 0
    ["R_thumb_proximal_yaw_joint", "right_thumb_1_joint"],
    ["R_thumb_proximal_pitch_joint", "right_thumb_2_joint"],
    ["R_thumb_intermediate_joint", "right_thumb_3_joint"],
    ["R_thumb_distal_joint", "right_thumb_4_joint"],
    ["R_index_proximal_joint", "right_index_1_joint"],
    # 5
    ["R_index_intermediate_joint", "right_index_2_joint"],
    ["R_middle_proximal_joint", "right_middle_1_joint"],
    ["R_middle_intermediate_joint", "right_middle_2_joint"],
    ["R_ring_proximal_joint", "right_ring_1_joint"],
    ["R_ring_intermediate_joint", "right_ring_2_joint"],
    # 10
    ["R_pinky_proximal_joint", "right_little_1_joint"],
    ["R_pinky_intermediate_joint", "right_little_2_joint"]
]

NAMES_JOINT_HAND_LEFT = []
NAMES_JOINT_HAND_RIGHT = []

if HAND == "Dex3-1":
    NAMES_JOINT_HAND_LEFT = NAMES_JOINT_DEX31_LEFT
    NAMES_JOINT_HAND_RIGHT = NAMES_JOINT_DEX31_RIGHT
    JOINT_NUM = 7
else:
    NAMES_JOINT_HAND_LEFT = NAMES_JOINT_INSPIREDHAND_LEFT
    NAMES_JOINT_HAND_RIGHT = NAMES_JOINT_INSPIREDHAND_RIGHT
    JOINT_NUM = 12
