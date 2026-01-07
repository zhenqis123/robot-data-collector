import pickle as pkl
import numpy as np
import json
import copy
import collections

import glob
import math
import matplotlib.pyplot as plt


from collections import defaultdict

import sys
import os


from typing import List, Dict, Union, Tuple, Any

from pathlib import Path

from glove_hand import *

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

CONTROL_FINGER_RATE = 0.05
MANO_VERTICES = 891 
GLOVE_TOTAL_DIM = 137

BEND_corners={
    # 从左上角到左下角到右下角到右上角
    "bend_thumb":[
        865,
        140,
        300,
        866
    ],
    "bend_index":[
        263,
        190,
        157,
        264
    ],
    "bend_middle":
    [
        452,
        429,
        421,
        460
    ],
    "bend_ring":
    [
        583,
        560,
        668,
        592
    ],
    "bend_pinky":
    [
        780,
        708,
        694,
        736
    ]
}


UV_MAPPING_CONFIG_W_BEND = [
    {
        "name": "palm",
        "sensor_shape": (5, 15),  # 假设手掌传感器是 15x5
        "source_points": np.array(
            [
                [0, 0],    # 左上
                [0, 14],   # 右上
                [4, 14],   # 右下
                [4, 0]     # 左下
            ],
            dtype=np.float32
        ),
        # 这是在UV图上精确查找的点的序号
        "target_indices": np.array(
            [145, 883, 55, 50],  # 对应左上、右上、右下、左下
            dtype=np.int32
        )
    },
    {
        "name": "thumb_tip",
        "sensor_shape": (4, 3),  # 假设拇指指尖传感器是 4x3
        "source_points": np.array(
            [
                [0, 0], [0, 2], [3, 2], [3, 0]  # 右上角 x 坐标从 3 改为 2
            ],
            dtype=np.float32
        ),
        "target_indices": np.array(
            # [844, 862, 870, 845],
            # fix the same point
            [843, 862, 870, 845],
            dtype=np.int32
        )
    },
    {
        "name": "index_tip",
        "sensor_shape": (4, 3),  # 假设食指指尖传感器是 4x3
        "source_points": np.array(
            [
                [0, 0], [0, 2], [3, 2], [3, 0]  # 右上角 x 坐标从 3 改为 2
            ],
            dtype=np.float32
        ),
        "target_indices": np.array(
            # [367, 398, 391, 374],
            # fix the same point
            [366, 398, 391, 374],
            dtype=np.int32
        )
    },
    {
        "name": "middle_tip",
        "sensor_shape": (4, 3),  # 假设中指指尖传感器是 4x3
        "source_points": np.array(
            [
                [0, 0], [0, 2], [3, 2], [3, 0]
            ],
            dtype=np.float32
        ),
        "target_indices": np.array(
            # [498, 530, 472, 505],
            [497, 530, 472, 505],
            dtype=np.int32
        )
    },
    {
        "name": "ring_tip",
        "sensor_shape": (4, 3),  # 假设无名指指尖传感器是 4x3
        "source_points": np.array(
            [
                [0, 0], [0, 2], [3, 2], [3, 0]
            ],
            dtype=np.float32
        ),
        "target_indices": np.array(
            [633, 662, 655, 637],
            dtype=np.int32
        )
    },
    {
        "name": "little_tip",
        "sensor_shape": (4, 3),  # 假设小拇指指尖传感器是 4x3
        "source_points": np.array(
            [
                [0, 0], [0, 2], [3, 2], [3, 0]
            ],
            dtype=np.float32
        ),
        "target_indices": np.array(
            [763, 792, 785, 768],
            dtype=np.int32
        )
    },
    {
        "name": "bend_thumb",
        "sensor_shape":(1,1),
        "source_points": np.array(
            [
                [0,0],
                [0,0],
                [0,0],
                [0,0]
            ],
            dtype = np.float32
        ),
        "target_indices" : np.array(
                    [
                865,
                140,
                300,
                866
            ],
        )
    },
    {
        "name": "bend_index",
        "sensor_shape":(1,1),
        "source_points": np.array(
            [
                [0,0],
                [0,0],
                [0,0],
                [0,0]
            ],
            dtype = np.float32
        ),
        "target_indices" : np.array(
                    [
                263,
                191,
                157,
                264
            ],
        )
    },
    {
        "name": "bend_middle",
        "sensor_shape":(1,1),
        "source_points": np.array(
            [
                [0,0],
                [0,0],
                [0,0],
                [0,0]
            ],
            dtype = np.float32
        ),
        "target_indices" : np.array(
                    [
                453,
                429,
                430,
                460
            ],
        )
    },
    {
        "name": "bend_ring",
        "sensor_shape":(1,1),
        "source_points": np.array(
            [
                [0,0],
                [0,0],
                [0,0],
                [0,0]
            ],
            dtype = np.float32
        ),
        "target_indices" : np.array(
                    [
                583,
                561,
                668,
                584
            ],
        )
    },
    {
        "name": "bend_pinky",
        "sensor_shape":(1,1),
        "source_points": np.array(
            [
                [0,0],
                [0,0],
                [0,0],
                [0,0]
            ],
            dtype = np.float32
        ),
        "target_indices" : np.array(
                    [
                717,
                708,
                707,
                718
            ],
        )
    },
]

UV_MAPPING_CONFIG_DICT_W_BEND = {item["name"]: item["sensor_shape"] for item in UV_MAPPING_CONFIG_W_BEND}




HAND_PALM_FIRST_INDEX = 65
HAND_VEC_INDICES_W_BEND ={
    "thumb": [0,1,2,3,4,5,6,7,8,9,10,11],
    "index": [13,14,15,16,17,18,19,20,21,22,23,24],
    "middle": [26,27,28,29,30,31,32,33,34,35,36,37],
    "ring": [39,40,41,42,43,44,45,46,47,48,49,50],
    "pinky": [52,53,54,55,56,57,58,59,60,61,62,63],
    "palm": list(range(HAND_PALM_FIRST_INDEX, 137)),
    "bend_thumb":[12],
    "bend_index":[25],
    "bend_middle":[38],
    "bend_ring":[51],
    "bend_pinky":[64],
}

HAND_VEC_NAME_TO_DICT_NAME_W_BEND = {
    "thumb": "thumb_tip",
    "index": "index_tip",
    "middle": "middle_tip",
    "ring": "ring_tip",
    "pinky": "little_tip",
    "palm": "palm",
    "bend_thumb":"bend_thumb",
    "bend_index":"bend_index",
    "bend_middle":"bend_middle",
    "bend_ring":"bend_ring",
    "bend_pinky":"bend_pinky",
}

HAND_BEND_INDICE = [12,25,38,51,64]  # thumb to pinky
HAND_PALM_FIRST_INDEX = 65

HAND_ALL_INDEX = 137

HAND_VEC_ASYMMETRIC_INDICES = {
    "thumb": [0,1,2,3,4,5,6,7,8,9,10,11],
    "index": [13,14,15,16,17,18,19,20,21,22,23,24],
    "middle": [26,27,28,29,30,31,32,33,34,35,36,37],
    "ring": [39,40,41,42,43,44,45,46,47,48,49,50],
    "pinky": [52,53,54,55,56,57,58,59,60,61,62,63],
    "palm": [68,69,70,71,72,73,74,75,76] + list(range(80, 137)),
    "bend": HAND_BEND_INDICE
}

HAND_VEC_INDICES = {
    "thumb": [0,1,2,3,4,5,6,7,8,9,10,11],
    "index": [13,14,15,16,17,18,19,20,21,22,23,24],
    "middle": [26,27,28,29,30,31,32,33,34,35,36,37],
    "ring": [39,40,41,42,43,44,45,46,47,48,49,50],
    "pinky": [52,53,54,55,56,57,58,59,60,61,62,63],
    "palm": list(range(HAND_PALM_FIRST_INDEX, 137)),
    "bend": HAND_BEND_INDICE
}

HAND_VEC_NAME_TO_DICT_NAME = {
    "thumb": "thumb_tip",
    "index": "index_tip",
    "middle": "middle_tip",
    "ring": "ring_tip",
    "pinky": "little_tip",
    "palm": "palm",
    "bend": None,
}

HAND_PALM_CORNER_INDICES = {
    "right":[65,66,67],
    "left":[74,75,76]
}

HAND_PALM_NAN_INDICES = {
    "left":[65,66,67],
    "right":[74,75,76]
}


OLD_LEFT_ITEM_LABELS_DICT = {
    "empty": 0,
    # "screwdriver": 1,
    "peach": 1,
    "tennis": 2,
    # "cube": 3,
    "lime": 3,
    "apple": 4,
    "banana": 6,
    "orange":7,
    "pear": 8,
    # "strawberry": 9,
    "plum": 5,
    "rounded_square_prism": 9,
}
OLD_LEFT_ITEM_LABELS = list(OLD_LEFT_ITEM_LABELS_DICT.keys())

OLD_ITEM_LABELS_DICT = {
    "empty": 0,
    # "screwdriver": 1,
    "pear": 1,
    "tuna can": 2,
    # "cube": 3,
    "lime": 3,
    "baseball": 4,
    "drill": 5,
    "banana": 6,
    "listerine":7,
    "cleanser": 8,
    # "strawberry": 9,
}

ITEM_LABELS_DICT = {
    "empty": 0,
    # "screwdriver": 1,
    "pear": 1,
    "tuna can": 2,
    # "cube": 3,
    "lime": 3,
    "baseball": 4,
    "drill": 5,
    "banana": 6,
    "listerine":7,
    # "cleanser": 8,
    # "strawberry": 9,
}

NEW_ITEM_LABELS_DICT = {
    "empty": 0,
    "coke": 1,
    "HFS": 2,
    "pear": 3,
    "strawberry": 4,
    "cube": 5,
    "soft": 6,
    "baseball": 7,
    "VC": 8,
    "YHP": 9
}

NEW_ITEM_LABELS = [
    "empty",
    "coke",
    "HFS",
    "pear",
    "strawberry",
    "cube",
    "soft",
    "baseball",
    "VC",
    "YHP"
]

NEWEST_ITEM_LABELS=[
    "empty",
    "RedBull",
    "SmallCube",
    "Charger",
    "Soft",
    "YSP",
    "Plum",
    "Apple",
    "Bottle",
    "Cylinder"
]

DEMO_ITEM_LABELS_DICT = {
    "empty": 0,
    # "screwdriver": 1,
    # "peach": 1,
    # "tennis": 2,
    # "cube": 3,
    # "lime": 3,
    "apple": 1,
    "rounded_square_prism": 2,
    "banana": 3,
    # "orange":7,
    # "pear": 8,
    # "strawberry": 9,
    "plum": 4,
}

PAIRED_ITEM_LABELS_DICT = [{
    "empty": 0,
    "softball":1,
    "coke":2,
    "tennis":3,
    "apple":4,
    "3D_cup":5,
    "nongfu_bottle":6,
    "VC":7,
    "mouse":8,
    "salt_bottle":9,
},{
    "cube": 1,
    "strawberry":2
}
]

WEIGHT_LABEL_DICT = {
    "empty_NF":0,
    "water_NF":1,
    "empty_JJ":2,
    "water_JJ":3,
    "empty_coconut":4,
    "water_coconut":5,
    "orange_soft":6,
    "orange_hard":7,
    "YHP_soft":8,
    "YHP_hard":9
}

WEIGHT_ITEM_LABELS = list(WEIGHT_LABEL_DICT.keys())

HAND_LABELS_DICT = {
    "thumb": 0,
    "index": 1,
    "middle": 2,
    "ring": 3,
    "pinky": 4,
    "palm": 5,
    "bend": 6
}

ACTION_DICT = {
    "still":0,
    "left": 1,
    "right": 2,
    "forward":3,
    "backward":4,
}

CLASSIFY_TYPE_DICT = {
    "item_label":0,
    "finger_contact_label":1,
    "paired_item_1":2, # paired item _1
    "paired_item_2":3, # paired item _2
    "action_label":4,
    "new_item_label":5,
    "paired_data":6,
    "newest_item_label":7,
    "weight_item_label":8
}
CLASSIFY_TYPE_LABELS = list(CLASSIFY_TYPE_DICT.keys())

def get_classify_type_name(type_str):
    if isinstance(type_str, int):
        type_str = CLASSIFY_TYPE_LABELS[type_str]
    if type_str == "action_label":
        return "action_label"
    if type_str == "new_item_label" or type_str == "newest_item_label" or type_str == "weight_item_label":
        return "item_label"
    if type_str != "paired_item_1" and type_str != "paired_item_2":
        return type_str
    else:
        return "item_label"

ACTION_LABELS = list(ACTION_DICT.keys())
ITEM_LABELS = list(ITEM_LABELS_DICT.keys())
PAIRED_ITEM_LABELS = [list(PAIRED_ITEM_LABELS_DICT[i].keys()) for i in range(len(PAIRED_ITEM_LABELS_DICT))]
PAIRED_ITEM_LABELS_1D =[key for d in PAIRED_ITEM_LABELS_DICT for key in d.keys()]
DEMO_ITEM_LABELS = list(DEMO_ITEM_LABELS_DICT.keys())

INSPIRE_CORRECT_ORD_RIGHT = None
INSPIRE_CORRECT_ORD_LEFT = None
HAND_LABELS = list(HAND_LABELS_DICT.keys())

# ITEM_LABELS_DICT = OLD_LEFT_ITEM_LABELS_DICT
# ITEM_LABELS = list(ITEM_LABELS_DICT.keys())


def get_type_label_list(type_mode):
    if isinstance(type_mode, int):
        type_mode = CLASSIFY_TYPE_LABELS[type_mode]
    if type_mode == "action_label":
        return ACTION_LABELS
    elif type_mode == "item_label":
        return ITEM_LABELS
    elif type_mode == "paired_item_1" or type_mode == "paired_item_2":
        return PAIRED_ITEM_LABELS_1D
    elif type_mode == "finger_contact_label":
        return HAND_LABELS
    elif type_mode == "new_item_label":
        return NEW_ITEM_LABELS
    elif type_mode == "newest_item_label":
        return NEWEST_ITEM_LABELS
    elif type_mode == "weight_item_label":
        return WEIGHT_ITEM_LABELS
    else:
        raise ValueError(f"Unknown type mode: {type_mode}")

HAND_INSPIRE_LABELS = {
    "thumb": ["thumb_1", "thumb_2", "thumb_3", ],
    "index": ["index_1", "index_2"],
    "middle": ["middle_1", "middle_2"],
    "ring": ["ring_1", "ring_2"],
    "pinky": ["little_1", "little_2"],
    "palm": ["palm"],
    "bend": [],
}

OLD_HAND_INSPIRE_ACTIVE_LABELS = {
    "thumb": ["thumb_1", "thumb_2", "thumb_3", ],
    "index": ["index_1", "index_2", "index_3"],
    "middle": ["middle_1", "middle_2", "middle_3"],
    "ring": ["ring_1", "ring_2", "ring_3"],
    "pinky": ["little_1", "little_2", "little_3"],
    "palm": ["palm"],
    "bend": [],
}

HAND_INSPIRE_ACTIVE_LABELS = {
    "thumb": ["thumb_2",],
    "index": ["index_2",],
    "middle": ["middle_2"],
    "ring": ["ring_2",],
    "pinky": ["little_2",],
    "palm": ["palm"],
    "bend": [],
}

OPERATOR2AVP_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)
OPERATOR2AVP_LEFT = np.array(
    [
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ]
)

TARGET_LINK_HUMAN_INDICES= np.array([
    [8, 12, 16, 20, 12, 16, 20, 16, 20, 20, 0, 0, 0, 0, 0],
    [4, 4, 4, 4, 8, 8, 8, 12, 12, 16, 4, 8, 12, 16, 20]
])


# 定义2D形状和需要填充0的位置
GLOVE_137_SHAPES = {
    "right":{
    "palm":  {"shape": (5, 15), "nan_indices": [(0, 0), (0, 1), (0, 2)]},
    "thumb": {"shape": (4, 3), "nan_indices": []},
    "index": {"shape": (4, 3), "nan_indices": []},
    "middle":{"shape": (4, 3), "nan_indices": []},
    "ring":  {"shape": (4, 3), "nan_indices": []},
    "pinky": {"shape": (4, 3), "nan_indices": []},
    # "bend" 是1D的，不需要在这里定义
    }
    , "left":{
    "palm":  {"shape": (5, 15), "nan_indices": [(0, 12), (0, 13), (0, 14)]},
    "thumb": {"shape": (4, 3), "nan_indices": []},
    "index": {"shape": (4, 3), "nan_indices": []},
    "middle":{"shape": (4, 3), "nan_indices": []},
    "ring":  {"shape": (4, 3), "nan_indices": []},
    "pinky": {"shape": (4, 3), "nan_indices": []},
    }
}

# --- 2. Inspire 手套的映射 (Mapping for the Inspire Glove) ---
INSPIRE_TAC_REGION_NAMES = ["little_1", "little_2", "little_3",
    "ring_1", "ring_2", "ring_3",
    "middle_1", "middle_2", "middle_3",
    "index_1", "index_2", "index_3",
    "thumb_1", "thumb_2", "thumb_3", "thumb_4",
    "palm"]
INSPIRE_TAC_REGION_SIZES = [(3,3), (12,8), (10,8),
    (3,3), (12,8), (10,8),
    (3,3), (12,8), (10,8),
    (3,3), (12,8), (10,8),
    (3,3), (12,8), (3,3), (12,8),
    (8,14)]


UV_MAPPING_CONFIG = [
    {
        "name": "palm",
        "sensor_shape": (5, 15),  # 假设手掌传感器是 15x5
        "source_points": np.array(
            [
                [0, 0],    # 左上
                [0, 14],   # 右上
                [4, 14],   # 右下
                [4, 0]     # 左下
            ],
            dtype=np.float32
        ),
        # 这是在UV图上精确查找的点的序号
        "target_indices": np.array(
            [145, 883, 55, 50],  # 对应左上、右上、右下、左下
            dtype=np.int32
        )
    },
    {
        "name": "thumb_tip",
        "sensor_shape": (4, 3),  # 假设拇指指尖传感器是 4x3
        "source_points": np.array(
            [
                [0, 0], [0, 2], [3, 2], [3, 0]  # 右上角 x 坐标从 3 改为 2
            ],
            dtype=np.float32
        ),
        "target_indices": np.array(
            # [844, 862, 870, 845],
            # fix the same point
            [843, 862, 870, 845],
            dtype=np.int32
        )
    },
    {
        "name": "index_tip",
        "sensor_shape": (4, 3),  # 假设食指指尖传感器是 4x3
        "source_points": np.array(
            [
                [0, 0], [0, 2], [3, 2], [3, 0]  # 右上角 x 坐标从 3 改为 2
            ],
            dtype=np.float32
        ),
        "target_indices": np.array(
            # [367, 398, 391, 374],
            # fix the same point
            [366, 398, 391, 374],
            dtype=np.int32
        )
    },
    {
        "name": "middle_tip",
        "sensor_shape": (4, 3),  # 假设中指指尖传感器是 4x3
        "source_points": np.array(
            [
                [0, 0], [0, 2], [3, 2], [3, 0]
            ],
            dtype=np.float32
        ),
        "target_indices": np.array(
            # [498, 530, 472, 505],
            [497, 530, 472, 505],
            dtype=np.int32
        )
    },
    {
        "name": "ring_tip",
        "sensor_shape": (4, 3),  # 假设无名指指尖传感器是 4x3
        "source_points": np.array(
            [
                [0, 0], [0, 2], [3, 2], [3, 0]
            ],
            dtype=np.float32
        ),
        "target_indices": np.array(
            [633, 662, 655, 637],
            dtype=np.int32
        )
    },
    {
        "name": "little_tip",
        "sensor_shape": (4, 3),  # 假设小拇指指尖传感器是 4x3
        "source_points": np.array(
            [
                [0, 0], [0, 2], [3, 2], [3, 0]
            ],
            dtype=np.float32
        ),
        "target_indices": np.array(
            [763, 792, 785, 768],
            dtype=np.int32
        )
    },
]

UV_MAPPING_CONFIG_DICT = {item["name"]: item["sensor_shape"] for item in UV_MAPPING_CONFIG}

# 自动生成Inspire手套的映射字典
INSPIRE_MAPPING = {}
_current_pos = 0
for name, shape in zip(INSPIRE_TAC_REGION_NAMES, INSPIRE_TAC_REGION_SIZES):
    num_elements = shape[0] * shape[1]
    INSPIRE_MAPPING[name] = {
        "slice": slice(_current_pos, _current_pos + num_elements),
        "shape": shape
    }
    _current_pos += num_elements

# Inspire 手套的总维度
INSPIRE_TOTAL_DIM = _current_pos


# 1. 旋转 (Abduction) 角度范围 (单位: 弧度)
#    拇指完全并拢(adducted)时的角度，需要校准
THUMB_ROTATION_MIN_ANGLE = 1.1
#    拇指完全张开(abducted)时的角度，需要校准
THUMB_ROTATION_MAX_ANGLE = 1.3

# 2. 收缩 (Flexion) 角度范围 (单位: 弧度)
#    拇指完全弯曲(flexed)时的角度
THUMB_CONTRACTION_MIN_ANGLE = 0.3
#    拇指完全伸直(extended)时的角度
THUMB_CONTRACTION_MAX_ANGLE = 1.0 # 接近 PI (180度)

# 3. 平滑因子 (0到1之间)
#    值越小，平滑效果越强，但响应越慢；值越大，响应越快，但容易抖动
SMOOTHING_FACTOR = 1

def calculate_thumb_articulation(mano_points, last_smooth_rotation=0.5, last_smooth_contraction=0.5):
    """
    计算大拇指的旋转和收缩值
    :param mano_points: numpy array, shape (21, 3)，包含21个关键点的xyz坐标
    :return: (rotation, contraction)，两个范围在[0, 1]的浮点数
    """

    # --- 步骤1: 提取关键点并计算向量 ---
    p = mano_points
    # 坐标系构建向量
    wrist_to_index = p[5] - p[0]
    wrist_to_pinky = p[17] - p[0]
    
    # 大拇指向量
    thumb_cmc_to_mcp = p[2] - p[0] # 拇指掌骨
    thumb_mcp_to_ip = p[3] - p[1]   # 近端指骨
    thumb_ip_to_tip = p[4] - p[3]   # 远端指骨

    # --- 步骤2: 构建稳定的手掌局部坐标系 ---
    # 手掌的 Y 轴 (法线方向, 掌心向上)
    palm_normal = np.cross(wrist_to_index, wrist_to_pinky)
    palm_y_axis = palm_normal / np.linalg.norm(palm_normal)
    
    # 手掌的 X 轴 (大致指向食指方向)
    palm_x_axis = wrist_to_index / np.linalg.norm(wrist_to_index)
    
    # 手掌的 Z 轴 (通过叉乘得到, 保证坐标系正交)
    palm_z_axis = np.cross(palm_x_axis, palm_y_axis)
    palm_z_axis /= np.linalg.norm(palm_z_axis)


    # --- 步骤3: 计算旋转 (Rotation/Abduction) ---
    # 将拇指掌骨向量投影到手掌平面 (XZ平面)
    thumb_metacarpal_proj = thumb_cmc_to_mcp - (np.dot(thumb_cmc_to_mcp, palm_y_axis) * palm_y_axis)
    
    # 计算投影向量与手掌 Z 轴的夹角
    # 使用 atan2 来获得带方向的、-PI到PI范围内的角度，更稳定
    angle_rad_rotation = np.arctan2(np.dot(thumb_metacarpal_proj, palm_x_axis), 
                                     np.dot(thumb_metacarpal_proj, palm_z_axis))
    # print("_____________________")
    # print(angle_rad_rotation)
    # print("_____________________")

    # --- 步骤4: 计算收缩 (Contraction/Flexion) ---
    # 计算近端指骨和远端指骨之间的夹角
    # 使用点积公式: A·B = |A||B|cos(theta)
    dot_product = np.dot(thumb_mcp_to_ip, thumb_ip_to_tip)
    mag_prod = np.linalg.norm(thumb_mcp_to_ip) * np.linalg.norm(thumb_ip_to_tip)
    # clip防止因计算误差导致值超出[-1, 1]范围
    angle_rad_contraction = np.arccos(np.clip(dot_product / mag_prod, -1.0, 1.0))
    # print("+++++++++++++++++++++")
    # print(angle_rad_contraction)
    # print("+++++++++++++++++++++")


    # --- 步骤5: 归一化与平滑 ---
    # 归一化旋转值
    rotation_raw = (angle_rad_rotation - THUMB_ROTATION_MIN_ANGLE) / (THUMB_ROTATION_MAX_ANGLE - THUMB_ROTATION_MIN_ANGLE)
    rotation_clamped = np.clip(rotation_raw, 0.0, 1.0)
    # 用户定义: 1是完全张开, 0是完全收缩。我们的计算结果是角度越大越张开，所以直接使用。
    rotation = rotation_clamped

    # 归一化收缩值
    contraction_raw = (angle_rad_contraction - THUMB_CONTRACTION_MIN_ANGLE) / (THUMB_CONTRACTION_MAX_ANGLE - THUMB_CONTRACTION_MIN_ANGLE)
    contraction_clamped = np.clip(contraction_raw, 0.0, 1.0)
    # 用户定义: 1是完全张开(伸直), 0是完全收缩(弯曲)。
    # 我们的计算是角度越大越伸直, 所以这个值正好符合要求。
    contraction = contraction_clamped
    
    # 应用指数移动平均 (EMA) 进行平滑
    smooth_rotation = (rotation * SMOOTHING_FACTOR) + (last_smooth_rotation * (1 - SMOOTHING_FACTOR))
    smooth_contraction = (contraction * SMOOTHING_FACTOR) + (last_smooth_contraction * (1 - SMOOTHING_FACTOR))

    # 更新上一帧的值
    last_smooth_rotation = smooth_rotation
    last_smooth_contraction = smooth_contraction
    
    return smooth_rotation, smooth_contraction, last_smooth_rotation, last_smooth_contraction

# ADJUSTMENT_SCALE_FACTORS_HIGH = [0.8, 0.8, 0.8, 0.8, 0.8]
# ADJUSTMENT_SCALE_FACTORS_LOW = [0.0, 0.0, 0.0, 0.0, 0.0]
# GLOVE_ACTIVE_VALUES = [7.0, 7.0, 2.0, 6.5]     
# INSPIRE_ACTIVE_VALUES = [0.25, 0.25, 0.25, 0.25]

# GLOVE_LOWER_THRESHOLDS = [0.1, 0.01, 0.1, 0.01, 0.05]
# GLOVE_UPPER_THRESHOLDS = [0.4, 0.4, 0.4, 0.4, 0.4]

ADJUSTMENT_SCALE_FACTORS_HIGH = [0.1, 0.1, 0.2, 0.2, 0.1] * 2
ADJUSTMENT_SCALE_FACTORS_LOW = [0.0, 0.0, 0.0, 0.0, 0.0]
GLOVE_ACTIVE_VALUES = [7.0, 7.0, 2.0, 6.5]     
INSPIRE_ACTIVE_VALUES = [0.25, 0.25, 0.25, 0.25]

GLOVE_LOWER_THRESHOLDS = [0.1, 0.1, 0.1, 0.1, 0.1] * 3
GLOVE_UPPER_THRESHOLDS = [0.8, 0.8, 0.8, 0.8, 0.8] * 5

# 假设其他常量不变
# HAND_VEC_INDICES = {...}
# HAND_INSPIRE_ACTIVE_LABELS = {...}
# HAND_LABELS = [...]

class TAC_TEMP_CLASS:
    def __init__(self):
        self.ori_tac = np.empty(0)
        self.input_inspire_actions = np.empty(0)
        self.labels = np.empty(0)
        self.glove_keypoints = np.empty(0)

# --- 2. 更新函数接口，传入手指索引 ---
def get_glove_activation(glove_vector, finger_name, finger_index):
    """
    计算 Glove 传感器的归一化激活值。
    使用对应维度的 GLOVE_ACTIVE_VALUES 进行归一化。
    """
    glove_vector = glove_vector.reshape(-1)
    assert len(glove_vector) == 137
    
    active_value = GLOVE_ACTIVE_VALUES[finger_index]
    if active_value == 0:
        return 0.0
        
    raw_value = glove_vector[HAND_VEC_INDICES[finger_name]].sum()
    return raw_value / active_value

def get_inspire_activation(inspire_tac_dict, finger_name, finger_index):
    """
    计算 Inspire 传感器的归一化激活值。
    使用对应维度的 INSPIRE_ACTIVE_VALUES 进行归一化。
    """
    total_val = 0.
    total_num = 0.
    for tac_region in [inspire_tac_dict[region_name] for region_name in HAND_INSPIRE_ACTIVE_LABELS[finger_name]]:
        total_val += tac_region.sum()
        total_num += tac_region.size
    
    active_value = INSPIRE_ACTIVE_VALUES[finger_index]
    if total_num == 0 or active_value == 0:
        return 0.0
        
    avg_val = total_val / total_num
    return avg_val / active_value

# --- 3. 主函数中使用索引来获取对应的超参数 ---
def adjust_inspire_action_old(inspire_action, inspire_tac_dict, glove_tac_vector, lp_filter=None):
    """
    根据每个维度独立的超参数，平滑地调整动作。
    """
    adjust_hand_labels = HAND_LABELS[0:-2]
    
    # 使用 enumerate 同时获取手指名称和索引 (i)
    glove_activations = [get_glove_activation(glove_vector=glove_tac_vector, finger_name=f_name, finger_index=i) 
                         for i, f_name in enumerate(adjust_hand_labels)]
    inspire_activations = [get_inspire_activation(inspire_tac_dict=inspire_tac_dict, finger_name=f_name, finger_index=i) 
                           for i, f_name in enumerate(adjust_hand_labels)]

    # 核心平滑逻辑
    for i in range(5):
        glove_act = glove_activations[i]
        inspire_act = inspire_activations[i]
        
        activation_diff = max(0, glove_act - inspire_act)
        
        # 使用对应维度的调整因子
        # scale_factor = ADJUSTMENT_SCALE_FACTORS_LOW[i] + inspire_action[3-i] * (ADJUSTMENT_SCALE_FACTORS_HIGH[i] - ADJUSTMENT_SCALE_FACTORS_LOW[i])
        adjustment = activation_diff

        # use ADJUSTMENT_SCALE_FACTORS to clip the adjustment

        
        # 应用调整
        if i ==0:
            inspire_action[4] -= adjustment
        else:
            inspire_action[4-i] -= adjustment
    
    if lp_filter is not None:
        inspire_action = lp_filter.next(inspire_action)
    
    return inspire_action


# --- 3. 主调整函数 (adjust_inspire_action) 的修改版 ---
def adjust_inspire_action(inspire_action, glove_tac_vector, lp_filter=None):
    """
    根据手套触觉的上下限阈值，平滑地调整机械手动作。
    此版本不再需要 inspire_tac_dict。
    """
    # 假设 HAND_LABELS 是一个已定义的列表
    adjust_hand_labels = HAND_LABELS[0:-2]
    
    # # 步骤 1: 计算所有相关手指的手套激活值
    # glove_activations = [get_glove_activation(glove_vector=glove_tac_vector, finger_name=f_name, finger_index=i) 
    #                      for i, f_name in enumerate(adjust_hand_labels)]

    glove_activations =   [glove_tac_vector[HAND_VEC_INDICES[finger_name]].mean() for finger_name in adjust_hand_labels]
    # print(max(glove_activations))
    # 步骤 2: 核心调整逻辑，仅依赖于手套激活值
    for i in range(5): # 假设处理4个手指
        glove_act = glove_activations[i]
        lower_thresh = GLOVE_LOWER_THRESHOLDS[i]
        upper_thresh = GLOVE_UPPER_THRESHOLDS[i]

        # 根据手套激活值和阈值计算一个0到1的触发因子 (adjustment_trigger)
        if glove_act <= lower_thresh:
            adjustment_trigger = 0.0
        elif glove_act >= upper_thresh:
            adjustment_trigger = 1.0
        else:
            # 在阈值区间内进行线性插值
            if upper_thresh > lower_thresh:
                adjustment_trigger = (glove_act - lower_thresh) / (upper_thresh - lower_thresh)
            else: # 避免除以零
                adjustment_trigger = 0.0
        
        adjustment_trigger = np.clip(adjustment_trigger, 0.0, 1.0)
        # print(adjustment_trigger)
        
        # 步骤 3: 计算最终的调整量
        # 动态调整因子 (scale_factor) 的逻辑保持不变，它使得调整力度与当前动作幅度相关
        scale_factor = ADJUSTMENT_SCALE_FACTORS_LOW[i] + adjustment_trigger * (ADJUSTMENT_SCALE_FACTORS_HIGH[i] - ADJUSTMENT_SCALE_FACTORS_LOW[i])
        
        adjustment = scale_factor

        # print(adjustment)
        
        # 应用调整
        # if i == 0:
        #     inspire_action[4] -= adjustment
        # else:

        inspire_action[4-i] -= adjustment

    # 步骤 5: 应用低通滤波器（可选）
    if lp_filter is not None:
        inspire_action = lp_filter.next(inspire_action)
        
    return inspire_action






def get_epoch_path(model_label, epoch, model_dir=None):
    if model_dir is None:
        model_dir = "./models"
    model_sub_dir = os.path.join(model_dir, model_label)
    if not os.path.exists(model_sub_dir):
        os.makedirs(model_sub_dir, exist_ok=True)
    return os.path.join(model_sub_dir, f"epoch_{epoch}.pth")

def flat1d_from_frame(frame, hand_type = HandType.RIGHT, reverse_right_left = None):
    
    # print(hand_type)
    tac_vec = frame.data[:256]
    tac_vec = np.frombuffer(tac_vec, dtype=np.uint8)

    # Create configurations and data structures for selected hands
    hand_config = HandConfig(hand_type)
    finger_val_mats = [hand_config.extract_finger_val_mat(i, tac_vec) for i in range(5)]
    # print(finger_val_mats)
    palm_val_mat = [hand_config.extract_palm_val_mat(tac_vec)]

    if reverse_right_left is None:
        reverse_right_left = (hand_type == HandType.RIGHT)
    
    return tac1d_from_mats(finger_val_mats, palm_val_mat, reverse_right_left)



def flip_glove_1d(
    glove_data_1d: np.ndarray, 
    hand_type
) -> np.ndarray:
    """
    将137维的Glove向量转换为其2D矩阵表示，对每个矩阵执行左右翻转(np.fliplr)，
    然后将翻转后的矩阵重新展平为137维向量。

    Args:
        glove_data_1d (np.ndarray): 输入的1D向量, 形状 (137,)。
        hand_type (str, optional): 手的类型 ('left' or 'right')。

    Returns:
        np.ndarray: 经过2D翻转后重组的1D向量, 形状 (137,)。
    """
    if glove_data_1d.shape != (GLOVE_TOTAL_DIM,):
        raise ValueError(f"Input vector must have shape ({GLOVE_TOTAL_DIM},) but got {glove_data_1d.shape}")

    if isinstance(hand_type, HandType):
        hand_type = hand_type.name.lower()
    if hand_type not in GLOVE_137_SHAPES:
        raise ValueError(f"hand_type must be one of {list(GLOVE_137_SHAPES.keys())}")
    

    # 初始化一个新的1D向量来存储翻转后的数据
    flipped_vector = np.zeros_like(glove_data_1d)
    
    # 遍历所有定义的部分
    for part_name, indices in HAND_VEC_INDICES.items():
        
        # 提取这部分对应的1D数据
        part_data_1d = glove_data_1d[indices]
        
        if part_name == 'bend':
            # 'bend' 部分是1D的，没有左右翻转的概念，所以直接复制
            flipped_vector[indices] = part_data_1d
            continue
            
        # --- 2D 部分 ---
        
        # 1. 从常量中获取形状信息
        reversed_hand_type = "left" if hand_type == "right" else "right"
        shape_info = GLOVE_137_SHAPES[reversed_hand_type][part_name]
        shape_2d = shape_info["shape"]
        nan_indices = shape_info["nan_indices"]
        
        # 2. Vector -> Mat: 将1D数据填充到2D矩阵中
        part_2d = np.zeros(shape_2d, dtype=glove_data_1d.dtype)
        
        # 创建一个 "有效" 索引的掩码
        valid_indices = np.ones(shape_2d, dtype=bool)
        for r, c in nan_indices:
            valid_indices[r, c] = False
            
        part_2d[valid_indices] = part_data_1d
        
        # 3. [核心操作] 左右翻转2D矩阵
        flipped_part_2d = np.fliplr(part_2d)
        
        # 4. Mat -> Vector: 从翻转后的矩阵中提取有效数据
        assert hand_type == "right"
        # TODO: only support right revert to left, not both ways for now
        if part_name == 'palm':
        #     # 手掌部分有特殊的索引范围
            flipped_part_2d[0][3:12] = flipped_part_2d[0][6:15]
            
        flipped_part_data_1d = flipped_part_2d[valid_indices]
        
        # 5. 将翻转后的1D数据放回新向量的正确位置
        flipped_vector[indices] = flipped_part_data_1d


    # flipped_vector[65:74] = flipped_vector[68:77]
    # flipped_vector[65:74] = np.flip(glove_data_1d[65:74])

    return flipped_vector

def flip_inspire_1d(
    inspire_data_1d: np.ndarray, 
    hand_type
) -> np.ndarray:
    
    if isinstance(hand_type, HandType):
        hand_type = hand_type.name.lower()
    if inspire_data_1d.shape != (INSPIRE_TOTAL_DIM,):
        raise ValueError(f"Input vector must have shape ({INSPIRE_TOTAL_DIM},) but got {inspire_data_1d.shape}")
    if hand_type not in ['left', 'right']:
        raise ValueError("hand_type must be 'left' or 'right'")

    reversed_hand_type = HandType.LEFT if hand_type == "right" else HandType.RIGHT
    flipped_mats = original_inspire_tac_vec_to_dict(inspire_data_1d, hand_type=reversed_hand_type)
    flipped_vector = original_inspire_tac_dict_to_vec(flipped_mats, hand_type=hand_type, flattened=True)
    
    return flipped_vector

def tac1d_from_mats(finger_val_mats, palm_val_mat, reverse_left_right = False) -> np.ndarray:
    """
    合并手部数据并展平为一维向量，同时去除NaN值
    
    参数:
    finger_val_mats: 包含手指数据的数组列表
    palm_val_mat: 包含手掌数据的数组列表
    
    返回:
    一维向量，包含所有非NaN值
    """
    # 提取所有数据数组
    all_arrays = []
    if isinstance(finger_val_mats, np.ndarray):
        assert len(finger_val_mats.shape)==3
        finger_val_mats = list(finger_val_mats)
        print()
    if isinstance(palm_val_mat, np.ndarray):
        if len(palm_val_mat.shape) == 2:
            palm_val_mat = [palm_val_mat]
        else:
            assert len(palm_val_mat.shape)==3
            palm_val_mat = list(palm_val_mat)
    assert isinstance(finger_val_mats, list)
    assert isinstance(palm_val_mat,list)
    
    # TODO：完善翻转
    if reverse_left_right:

        # 添加手指数据
        for arr in finger_val_mats:
            all_arrays.append(np.flip(arr,axis=1))
        
        # 添加手掌数据
        for arr in palm_val_mat:
            all_arrays.append(np.flip(arr,axis=1))
    else:
    # 添加手指数据
        # print(finger_val_mats[0])
        for arr in finger_val_mats:
            all_arrays.append(arr)
        
        # 添加手掌数据
        for arr in palm_val_mat:
            # print("palm_val_mat: ", arr)
            all_arrays.append(arr)

    # 合并所有数组并展平
    flattened = np.concatenate([_arr.flatten() for _arr in all_arrays])
    
    # 去除NaN值
    result = flattened[~np.isnan(flattened)]
    
    return result  

def glove_tac_dict_from_flat1d(
    flat_vec: np.ndarray,
    hand_type: HandType,):
    uv_glove_dict = {}
    for key, value in HAND_VEC_NAME_TO_DICT_NAME.items():
        if value:
            tac_value = flat_vec[HAND_VEC_INDICES[key]]
            if key == 'palm':
                palm_pad_value = np.zeros(3)
                if hand_type == HandType.RIGHT:
                    tac_value = np.concatenate([palm_pad_value, tac_value])
                else:
                    tac_value = np.concatenate([tac_value[0:12], palm_pad_value, tac_value[12:]])
            tac_shape = UV_MAPPING_CONFIG_DICT[value]
            tac_value_2d = tac_value.reshape(tac_shape)
            if hand_type == HandType.LEFT:
                tac_value_2d = np.flip(tac_value_2d, axis=1)
            uv_glove_dict[value] = tac_value_2d
    return uv_glove_dict

def glove_tac_dict_from_flat_batch(
    flat_vec_batch: np.ndarray,
    hand_type: HandType,
    with_bend:bool = False) -> Dict[str, np.ndarray]:
    """
    将(N, 137)的扁平化向量批次转换为传感器矩阵字典。
    字典中的每个值都是一个(N, m, n)的矩阵批次。
    """
    N = flat_vec_batch.shape[0]
    uv_glove_dict = {}
    if not with_bend:
        for key, value in HAND_VEC_NAME_TO_DICT_NAME.items():
            if value:
                # 沿第二个维度对整个批次进行切片
                tac_value = flat_vec_batch[:, HAND_VEC_INDICES[key]]
                
                if key == 'palm':
                    # 为批次中的每个项目进行填充
                    palm_pad_value = np.zeros((N, 3))
                    if hand_type == HandType.RIGHT:
                        tac_value = np.concatenate([palm_pad_value, tac_value], axis=1)
                    else:
                        tac_value = np.concatenate([tac_value[:, 0:12], palm_pad_value, tac_value[:, 12:]], axis=1)
                
                tac_shape = UV_MAPPING_CONFIG_DICT[value]
                # 对整个批次进行重塑
                tac_value_2d = tac_value.reshape(N, *tac_shape)

                if hand_type == HandType.LEFT:
                    # 沿列轴(对于批处理是axis=2)翻转
                    tac_value_2d = np.flip(tac_value_2d, axis=2)
                
                uv_glove_dict[value] = tac_value_2d
    else:
        for key, value in HAND_VEC_NAME_TO_DICT_NAME_W_BEND.items():
            if value:
                # 沿第二个维度对整个批次进行切片
                tac_value = flat_vec_batch[:, HAND_VEC_INDICES_W_BEND[key]]
                
                if key == 'palm':
                    # 为批次中的每个项目进行填充
                    palm_pad_value = np.zeros((N, 3))
                    if hand_type == HandType.RIGHT:
                        tac_value = np.concatenate([palm_pad_value, tac_value], axis=1)
                    else:
                        tac_value = np.concatenate([tac_value[:, 0:12], palm_pad_value, tac_value[:, 12:]], axis=1)
                
                tac_shape = UV_MAPPING_CONFIG_DICT_W_BEND[value]
                # 对整个批次进行重塑
                tac_value_2d = tac_value.reshape(N, *tac_shape)

                if hand_type == HandType.LEFT:
                    # 沿列轴(对于批处理是axis=2)翻转
                    tac_value_2d = np.flip(tac_value_2d, axis=2)
                
                uv_glove_dict[value] = tac_value_2d
    return uv_glove_dict

def original_inspire_tac_vec_to_dict(tac_vec, hand_type, flatten = True):
    n_tac_regions = 17
    tac_region_names = ["little_1", "little_2", "little_3",
                                "ring_1", "ring_2", "ring_3",
                                "middle_1", "middle_2", "middle_3",
                                "index_1", "index_2", "index_3",
                                "thumb_1", "thumb_2", "thumb_3", "thumb_4",
                                "palm"]
    tac_region_sizes = [(3,3), (12,8), (10,8),
                                (3,3), (12,8), (10,8),
                                (3,3), (12,8), (10,8),
                                (3,3), (12,8), (10,8),
                                (3,3), (12,8), (3,3), (12,8),
                                (8,14)]
    tac_region_start_idxs = [0] + [sum([s[0]*s[1] for s in tac_region_sizes[:i]]) for i in range(1, len(tac_region_sizes)+1)]

    tac_dict = {}
    for i in range(n_tac_regions):
        region_name = tac_region_names[i]
        region_size = tac_region_sizes[i]
        region_data = tac_vec[tac_region_start_idxs[i]:tac_region_start_idxs[i+1]].reshape(region_size)
        if region_name.endswith("_1") or region_name=="thumb_3" or region_name == 'palm':
            # 3*3 sensors and palm sensors are rotated 90 deg
            region_data = region_data.reshape(region_size[1], region_size[0]).T
        elif region_name=="thumb_4":
            # thumb_4 sensors are up-side-down
            region_data = region_data[::-1, ::-1]
            
        if hand_type==HandType.RIGHT:
            tac_dict[region_name] = region_data
        else:
            tac_dict[region_name] = np.flip(region_data, axis=1)
        
        if flatten:
            tac_dict[region_name] = tac_dict[region_name].flatten()

    return tac_dict

def original_inspire_tac_dict_to_vec(tac_dict, hand_type, flattened=True):
    """
    逆过程函数: 将触觉字典转换回原始的扁平向量。
    
    参数:
    - tac_dict (dict): 包含触觉区域数据的字典。
    - hand_type (HandType): 手的类型 (HandType.LEFT 或 HandType.RIGHT)。
    - flattened (bool): 字典中的值是否是扁平的 (即, 它们是否为1D向量)。
                           这应与原始函数的 `flatten` 参数匹配。
                           
    返回:
    - np.ndarray: 重新构建的 1D 触觉向量。
    """
    
    # --- 复制原始函数中的常量 ---
    n_tac_regions = 17
    tac_region_names = ["little_1", "little_2", "little_3",
                                "ring_1", "ring_2", "ring_3",
                                "middle_1", "middle_2", "middle_3",
                                "index_1", "index_2", "index_3",
                                "thumb_1", "thumb_2", "thumb_3", "thumb_4",
                                "palm"]
    tac_region_sizes = [(3,3), (12,8), (10,8),
                                (3,3), (12,8), (10,8),
                                (3,3), (12,8), (10,8),
                                (3,3), (12,8), (10,8),
                                (3,3), (12,8), (3,3), (12,8),
                                (8,14)]
    tac_region_start_idxs = [0] + [sum([s[0]*s[1] for s in tac_region_sizes[:i]]) for i in range(1, len(tac_region_sizes)+1)]
    # --- 常量结束 ---

    # 确定输出向量的总大小和数据类型
    total_size = tac_region_start_idxs[-1]
    
    # 从字典中的第一个元素获取 dtype
    try:
        first_key = tac_region_names[0]
        dtype = tac_dict[first_key].dtype
    except KeyError:
        raise ValueError(f"tac_dict 中缺少键: {first_key}")
    except Exception as e:
        print(f"无法从字典中确定 dtype: {e}")
        dtype = np.float32 # 默认值

    # 预先分配输出向量
    tac_vec = np.empty(total_size, dtype=dtype)
    
    # 严格按照原始顺序遍历以重建向量
    for i in range(n_tac_regions):
        region_name = tac_region_names[i]
        region_size = tac_region_sizes[i] # (rows, cols)
        
        # 1. 从字典中获取数据
        # 使用 .copy() 以避免修改原始字典
        region_data = tac_dict[region_name].copy() 

        # 2. [逆] un-flatten: 如果数据是扁平的, 将其重塑为 2D
        # 原始函数在所有变换之后才进行 flatten,
        # 并且所有变换都保持了 (rows, cols) 即 region_size 的形状。
        if flattened:
            region_data = region_data.reshape(region_size)

        # 3. [逆] Hand Type Flip: 如果是左手, 翻转回来
        # np.flip(axis=1) 是其自身的逆操作
        if hand_type == HandType.LEFT:
            region_data = np.flip(region_data, axis=1)
            
        # 4. [逆] Rotations / Flips
        if region_name.endswith("_1") or region_name=="thumb_3" or region_name == 'palm':
            # 原始操作: C = A.reshape(cols, rows).T
            # 逆操作: A = C.T.reshape(rows, cols)
            # A 的形状是 (rows, cols), 即 region_size
            region_data = region_data.T.reshape(region_size)
        elif region_name=="thumb_4":
            # 原始操作: A[::-1, ::-1]
            # 逆操作: A[::-1, ::-1] (其自身的逆操作)
            region_data = region_data[::-1, ::-1]
        
        # 5. [逆] Reshape & Slice Assignment
        # 此时, region_data 应该是 (rows, cols) 形状,
        # 与从 tac_vec 切片后立即得到的形状相同。
        # 将其扁平化并放回输出向量的正确位置。
        tac_vec[tac_region_start_idxs[i]:tac_region_start_idxs[i+1]] = region_data.flatten()

    return tac_vec

def seq_inspire_vec_from_dict(tac_dict):
    n_tac_regions = 17
    tac_region_names = ["little_1", "little_2", "little_3",
                                "ring_1", "ring_2", "ring_3",
                                "middle_1", "middle_2", "middle_3",
                                "index_1", "index_2", "index_3",
                                "thumb_1", "thumb_2", "thumb_3", "thumb_4",
                                "palm"]
    tac_region_sizes = [(3,3), (12,8), (10,8),
                                (3,3), (12,8), (10,8),
                                (3,3), (12,8), (10,8),
                                (3,3), (12,8), (10,8),
                                (3,3), (12,8), (3,3), (12,8),
                                (8,14)]
    tac_region_start_idxs = [0] + [sum([s[0]*s[1] for s in tac_region_sizes[:i]]) for i in range(1, len(tac_region_sizes)+1)]

    tac_vec = np.zeros(sum([s[0]*s[1] for s in tac_region_sizes]))
    for i, region_name in enumerate(tac_region_names):
        region_data = tac_dict[region_name]
        region_size = tac_region_sizes[i]
        tac_vec[tac_region_start_idxs[i]:tac_region_start_idxs[i+1]] = region_data.flatten()

    return tac_vec

def correct_inspire_vec_from_1d(tac_vec, hand_type):
    VECTOR_SIZE = 1062 # 触觉向量的固定长度
    # 断言检查输入的最后一个维度是否为1062
    assert tac_vec.shape[-1] == VECTOR_SIZE, \
        f"输入向量的最后一个维度必须是 {VECTOR_SIZE}, 但实际为 {tac_vec.shape[-1]}"

    global INSPIRE_CORRECT_ORD_RIGHT,INSPIRE_CORRECT_ORD_LEFT
    if hand_type==HandType.RIGHT:
        if INSPIRE_CORRECT_ORD_RIGHT is None:
            inspire_tac_input = np.zeros_like(tac_vec)
            for k in range(inspire_tac_input.shape[0]):
                inspire_tac_input[k] = k
            # print(inspire_tac_input)
            inspire_dict = original_inspire_tac_vec_to_dict(inspire_tac_input,hand_type=hand_type)
            INSPIRE_CORRECT_ORD_RIGHT = seq_inspire_vec_from_dict(inspire_dict)
            INSPIRE_CORRECT_ORD_RIGHT = INSPIRE_CORRECT_ORD_RIGHT.astype(np.int32)
            # breakpoint()
        return tac_vec[...,INSPIRE_CORRECT_ORD_RIGHT]
    else:
        if INSPIRE_CORRECT_ORD_LEFT is None:
            inspire_tac_input = np.zeros_like(tac_vec)
            for k in range(inspire_tac_input.shape[0]):
                inspire_tac_input[k] = k
            # print(inspire_tac_input)
            inspire_dict = original_inspire_tac_vec_to_dict(inspire_tac_input,hand_type=hand_type)
            INSPIRE_CORRECT_ORD_LEFT = seq_inspire_vec_from_dict(inspire_dict)
            INSPIRE_CORRECT_ORD_LEFT = INSPIRE_CORRECT_ORD_LEFT.astype(np.int32)
            # breakpoint()
        return tac_vec[...,INSPIRE_CORRECT_ORD_LEFT]

def mats_from_flat1d(flattened_vector, hand_type=HandType.LEFT):
    """
    将处理后的一维向量还原为原始的手部数据格式
    
    参数:
    flattened_vector: 处理后的一维向量（已去除NaN值）
    hand_type: 手型（LEFT/RIGHT），用于选择手掌模板
    
    返回:
    finger_val_mats: 还原后的手指数据
    palm_val_mat: 还原后的手掌数据
    """
    # 创建手指数据模板（5个手指，每个形状为(5,3)）
    finger_templates = [
        np.array([[0., 0., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.],
                  [np.nan, 0., np.nan]]),
        np.array([[0., 0., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.],
                  [np.nan, 0., np.nan]]),
        np.array([[0., 0., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.],
                  [np.nan, 0., np.nan]]),
        np.array([[0., 0., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.],
                  [np.nan, 0., np.nan]]),
        np.array([[0., 0., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.],
                  [np.nan, 0., np.nan]]),
    ]
    
    # 创建手掌数据模板（1个数组，形状为(5,15)）
    if hand_type == HandType.LEFT:
        palm_template = [
            np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., np.nan, np.nan, np.nan],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        ]
    else:  # 右手模板，修复形状错误（移除多余逗号）
        palm_template = [
            np.array([[np.nan, np.nan, np.nan, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        ]
    
    # 收集所有非NaN位置的索引（按原始展平顺序）
    all_indices = []
    
    # 处理手指数据模板（用enumerate直接获取索引，避免index()错误）
    for arr_idx, arr in enumerate(finger_templates):
        # 找到所有非NaN的位置（原始数据中这些位置有有效值）
        non_nan_mask = ~np.isnan(arr)
        non_nan_indices = np.where(non_nan_mask)
        # 记录位置坐标（手指类型、数组索引、行、列）
        for i, j in zip(non_nan_indices[0], non_nan_indices[1]):
            all_indices.append(('finger', arr_idx, i, j))
    
    # 处理手掌数据模板（同样用enumerate获取索引）
    for arr_idx, arr in enumerate(palm_template):
        non_nan_mask = ~np.isnan(arr)
        non_nan_indices = np.where(non_nan_mask)
        for i, j in zip(non_nan_indices[0], non_nan_indices[1]):
            all_indices.append(('palm', arr_idx, i, j))
    
    # 检查输入向量长度是否匹配预期的非NaN值数量
    if len(flattened_vector) != len(all_indices):
        raise ValueError(
            f"输入向量长度({len(flattened_vector)})与预期非NaN值数量({len(all_indices)})不匹配"
        )
    
    # 初始化结果数组（复制模板结构，保留NaN位置）
    finger_val_mats = [arr.copy() for arr in finger_templates]
    palm_val_mat = [arr.copy() for arr in palm_template]
    
    # 填充数据到非NaN位置
    for idx, val in enumerate(flattened_vector):
        data_type, arr_idx, i, j = all_indices[idx]
        if data_type == 'finger':
            finger_val_mats[arr_idx][i, j] = val
        else:  # palm
            palm_val_mat[arr_idx][i, j] = val
    
    return finger_val_mats, palm_val_mat



