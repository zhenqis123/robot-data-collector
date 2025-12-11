import platform
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PathConfig import PROJECT_ROOT, TACTASK_DIR, TELEOP_DIR, TACDATA_DIR

from ctypes import *
from collections import namedtuple
from teleop.VDGloves.vdmocapsdk_nodelist import LENGTH_BODY, LENGTH_HAND
from time import sleep


if platform.system() == "Windows":
    SDK_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            'VDMocapSDK_DataRead.dll')
else:
    SDK_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            'libVDMocapSDK_DataRead.so')
DATAREAD = CDLL(SDK_PATH)


SensorState = namedtuple('SensorState', [
    'SS_NONE',
    'SS_Well',
    'SS_NoData',
    'SS_UnReady',
    'SS_BadMag',
])._make(range(5))

WorldSpace = namedtuple('WorldSpace', [
    'WS_Geo',
    'WS_Unity',
    'WS_UE4',
])(0, 1, 2)

CharSet = namedtuple('CharSet', [
    'CHAR_AUTO',
    'CHAR_UTF5',
])(0, 1)

BvhFormat = namedtuple('BvhFormat', [
    'BVH_Biovision_BVH',
    'BVH_3ds_max_biped',
])(0, 1)


class MocapData(Structure):
    _fields_ = [("isUpdate", c_bool), ("frameIndex", c_uint),
                ("frequency", c_int), ("nsResult", c_int),
                ("sensorState_body", c_uint * LENGTH_BODY),
                ("position_body", c_float * 3 * LENGTH_BODY),
                ("quaternion_body", c_float * 4 * LENGTH_BODY),
                ("gyr_body", c_float * 3 * LENGTH_BODY),
                ("acc_body", c_float * 3 * LENGTH_BODY),
                ("velocity_body", c_float * 3 * LENGTH_BODY),
                ("sensorState_rHand", c_uint * LENGTH_HAND),
                ("position_rHand", c_float * 3 * LENGTH_HAND),
                ("quaternion_rHand", c_float * 4 * LENGTH_HAND),
                ("gyr_rHand", c_float * 3 * LENGTH_HAND),
                ("acc_rHand", c_float * 3 * LENGTH_HAND),
                ("velocity_rHand", c_float * 3 * LENGTH_HAND),
                ("sensorState_lHand", c_uint * LENGTH_HAND),
                ("position_lHand", c_float * 3 * LENGTH_HAND),
                ("quaternion_lHand", c_float * 4 * LENGTH_HAND),
                ("gyr_lHand", c_float * 3 * LENGTH_HAND),
                ("acc_lHand", c_float * 3 * LENGTH_HAND),
                ("velocity_lHand", c_float * 3 * LENGTH_HAND),
                ("isUseFaceBlendShapesARKit", c_bool),
                ("isUseFaceBlendShapesAudio", c_bool),
                ("faceBlendShapesARKit", c_float * 52),
                ("faceBlendShapesAudio", c_float * 26),
                ("localQuat_RightEyeball", c_float * 4),
                ("localQuat_LeftEyeball", c_float * 4),
                ("gestureResultL", c_int), ("gestureResultR", c_int)]


class Version(Structure):
    _fields_ = [("Project_Name", c_ubyte * 26),
                ("Author_Organization", c_ubyte * 128),
                ("Author_Domain", c_ubyte * 26),
                ("Author_Maintainer", c_ubyte * 26), ("Version", c_ubyte * 26),
                ("Version_Major", c_ubyte), ("Version_Minor", c_ubyte),
                ("Version_Patch", c_ubyte)]


def get_version(ver):
    DATAREAD.GetVerisonInfo(byref(ver))


def udp_set_position_in_initial_tpose(index, ip, port, ws, body, r_hand, l_hand):
    index = c_int(index)
    ip = create_string_buffer(ip.encode('gbk'), 30)
    port = c_ushort(port)
    ws = c_int(ws)
    type_float_array_3 = c_float * 3
    type_float_array_3_23 = type_float_array_3 * 23
    type_float_array_3_20 = type_float_array_3 * 20
    c_body = type_float_array_3_23()
    c_lhand = type_float_array_3_20()
    c_rhand = type_float_array_3_20()

    for i in range(len(body)):
        for j in range(len(body[0])):
            c_body[i][j] = c_float(body[i][j])
    for i in range(len(r_hand)):
        for j in range(len(r_hand[0])):
            c_rhand[i][j] = c_float(r_hand[i][j])
            c_lhand[i][j] = c_float(l_hand[i][j])

    DATAREAD.UdpSetPositionInInitialTpose.restype = c_bool
    res = DATAREAD.UdpSetPositionInInitialTpose(index, ip, port, ws, c_body,
                                                c_rhand, c_lhand)
    return bool(res)


def udp_get_recv_initial_tpose_position(index, ip, port, ws, body, r_hand, l_hand):
    index = c_int(index)
    ip = create_string_buffer(ip.encode('gbk'), 30)
    port = c_ushort(port)
    ws = c_int(ws)

    type_float_array_3 = c_float * 3
    type_float_array_3_23 = c_float * 3 * 23
    type_float_array_3_20 = c_float * 3 * 20
    c_body = type_float_array_3_23()
    c_rhand = type_float_array_3_20()
    c_lhand = type_float_array_3_20()

    for i in range(23):
        for j in range(len(body[0])):
            c_body[i][j] = c_float(body[i][j])
    for i in range(20):
        for j in range(len(r_hand[0])):
            c_rhand[i][j] = c_float(r_hand[i][j])
            c_lhand[i][j] = c_float(l_hand[i][j])
    DATAREAD.UdpGetRecvInitialTposePosition.restype = c_bool
    res = DATAREAD.UdpGetRecvInitialTposePosition(index, ip, port, ws, c_body,
                                                  c_rhand, c_lhand)
    sleep(0.1)
    for i in range(len(body)):
        for j in range(len(body[0])):
            body[i][j] = float(body[i][j])
    for i in range(len(r_hand)):
        for j in range(len(r_hand[0])):
            r_hand[i][j] = float(c_rhand[i][j])
            l_hand[i][j] = float(c_lhand[i][j])
    return bool(res)


def udp_open(index, port):
    index = c_int(index)
    DATAREAD.UdpOpen.restype = c_bool
    res = DATAREAD.UdpOpen(index, port)
    sleep(0.1)
    return bool(res)


def udp_close(index):
    sleep(0.1)
    index = c_int(index)
    DATAREAD.UdpClose(index)


def udp_is_open(index):
    index = c_int(index)
    DATAREAD.UdpIsOpen.restype = c_bool
    res = DATAREAD.UdpIsOpen(index)
    return bool(res)


def udp_remove(index, ip, port):
    index = c_int(index)
    ip = create_string_buffer(ip.encode('gbk'), 30)
    port = c_ushort(port)
    DATAREAD.UdpRemove.restype = c_bool
    res = DATAREAD.UdpRemove(index, ip, port)
    return bool(res)


def udp_send_request_connect(index, ip, port):
    index = c_int(index)
    ip = create_string_buffer(ip.encode('gbk'), 30)
    port = c_ushort(port)
    DATAREAD.UdpSendRequestConnect.restype = c_bool
    res = DATAREAD.UdpSendRequestConnect(index, ip, port)
    sleep(0.1)
    return bool(res)


def udp_recv_mocap_data(index, ip, port, mocap_data):
    index = c_int(index)
    ip = create_string_buffer(ip.encode('gbk'), 30)
    port = c_ushort(port)
    DATAREAD.UdpRecvMocapData.restype = c_bool
    res = DATAREAD.UdpRecvMocapData(index, ip, port, pointer(mocap_data))
    return bool(res)


def udp_recv_g1_data(index, ip, port, radians_current, radian_velocities_current,
                  radians_target, controls):
    type_float_arr29 = c_float * 29

    index = c_int(index)
    ip = create_string_buffer(ip.encode('gbk'), 30)
    port = c_ushort(port)
    c_radians_current = type_float_arr29(*radians_current)
    c_radian_velocities_current = type_float_arr29(*radian_velocities_current)
    c_radians_target = type_float_arr29(*radians_target)
    c_controls = type_float_arr29(*controls)

    DATAREAD.UdpRecvG1Data.restype = c_bool
    res = DATAREAD.UdpRecvG1Data(index, ip, port, c_radians_current,
                                 c_radian_velocities_current, c_radians_target,
                                 c_controls)

    for i in range(29):
        radians_target[i] = c_radians_target[i]
        controls[i] = c_controls[i]

    return bool(res)

def udp_recv_robot_data_dex3_1(index, ip, port, eulers_left, eulers_right):
    type_float_arr7 = c_float * 7

    index = c_int(index)
    ip = create_string_buffer(ip.encode('gbk'), 30)
    port = c_ushort(port)
    c_eulers_left = type_float_arr7()
    c_eulers_right = type_float_arr7()

    DATAREAD.UdpRecvRobotData_Dex3_1.restype = c_bool
    res = DATAREAD.UdpRecvRobotData_Dex3_1(index, ip, port, c_eulers_left,
                                           c_eulers_right)

    for i in range(7):
        eulers_left[i] = c_eulers_left[i]
        eulers_right[i] = c_eulers_right[i]

    return bool(res)

def udp_recv_robot_data_inspired_hand(index, ip, port, eulers_left, eulers_right):
    type_float_arr12 = c_float * 12

    index = c_int(index)
    ip = create_string_buffer(ip.encode('gbk'), 30)
    port = c_ushort(port)
    c_eulers_left = type_float_arr12()
    c_eulers_right = type_float_arr12()



    DATAREAD.UdpRecvRobotData_InspiredHand.restype = c_bool
    res = DATAREAD.UdpRecvRobotData_InspiredHand(index, ip, port, c_eulers_left,
                                           c_eulers_right)

    for i in range(12):
        eulers_left[i] = c_eulers_left[i]
        eulers_right[i] = c_eulers_right[i]

    return bool(res)
