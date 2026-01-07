#pragma once
#ifndef VDMOCAPSDK_DATAREAD_DATATYPE_H
#define VDMOCAPSDK_DATAREAD_DATATYPE_H

#define NODES_BODY 23
#define NODES_HAND 20
#define NODES_FACEBS_ARKIT 52
#define NODES_FACEBS_AUDIO 26
#define TRACKER_BODY 3 //光学追踪器身体部分数量, 分别是Hips/RightFoot/LeftFoot
#define BROADCAST_MAX_NUM  20                //最大支持20个模型广播
#define UPDATEID_MAX_ID    10                 //一次更新最大获取的Id数

namespace VDDataRead
{
	/**
	* @brief
	*   全身各节点及其序号。
	*/
	typedef enum BODYNODES
	{
		BN_Hips = 0,                    ///< Hips
		BN_RightUpperLeg,               ///< Right Upper Leg
		BN_RightLowerLeg,               ///< Right Lower Leg
		BN_RightFoot,                   ///< Right Foot
		BN_RightToe,                    ///< Right Toe
		BN_LeftUpperLeg,                ///< Left Upper Leg
		BN_LeftLowerLeg,                ///< Left Lower Leg
		BN_LeftFoot,                    ///< Left Foot
		BN_LeftToe,                     ///< Left Toe
		BN_Spine,                       ///< Spine
		BN_Spine1,                      ///< Spine1
		BN_Spine2,                      ///< Spine2
		BN_Spine3,                      ///< Spine3 -- Back
		BN_Neck,                        ///< Neck
		BN_Head,                        ///< Head
		BN_RightShoulder,               ///< Right Shoulder
		BN_RightUpperArm,               ///< Right Upper Arm
		BN_RightLowerArm,               ///< Right Lower Arm
		BN_RightHand,                   ///< Right Hand
		BN_LeftShoulder,                ///< Left Shoulder
		BN_LeftUpperArm,                ///< Left Upper Arm
		BN_LeftLowerArm,                ///< Left Lower Arm
		BN_LeftHand,                    ///< Left Hand
	}_BodyNodes_;



	/**
	* @brief
	*   Hand nodes name and its index.
	*/
	typedef enum HANDNODES
	{
		HN_Hand = 0,
		HN_ThumbFinger,
		HN_ThumbFinger1,
		HN_ThumbFinger2,
		HN_IndexFinger,
		HN_IndexFinger1,
		HN_IndexFinger2,
		HN_IndexFinger3,
		HN_MiddleFinger,
		HN_MiddleFinger1,
		HN_MiddleFinger2,
		HN_MiddleFinger3,
		HN_RingFinger,
		HN_RingFinger1,
		HN_RingFinger2,
		HN_RingFinger3,
		HN_PinkyFinger,
		HN_PinkyFinger1,
		HN_PinkyFinger2,
		HN_PinkyFinger3,
	}_HandNodes_;




	//Face BlendShape index
	typedef enum FACEBLENDSHAPEARKIT
	{
		ARKIT_BrowDownLeft = 0,
		ARKIT_BrowDownRight,
		ARKIT_BrowInnerUp,
		ARKIT_BrowOuterUpLeft,
		ARKIT_BrowOuterUpRight,
		ARKIT_CheekPuff,
		ARKIT_CheekSquintLeft,
		ARKIT_CheekSquintRight,
		ARKIT_EyeBlinkLeft,
		ARKIT_EyeBlinkRight,
		ARKIT_EyeLookDownLeft,
		ARKIT_EyeLookDownRight,
		ARKIT_EyeLookInLeft,
		ARKIT_EyeLookInRight,
		ARKIT_EyeLookOutLeft,
		ARKIT_EyeLookOutRight,
		ARKIT_EyeLookUpLeft,
		ARKIT_EyeLookUpRight,
		ARKIT_EyeSquintLeft,
		ARKIT_EyeSquintRight,
		ARKIT_EyeWideLeft,
		ARKIT_EyeWideRight,
		ARKIT_JawForward,
		ARKIT_JawLeft,
		ARKIT_JawOpen,
		ARKIT_JawRight,
		ARKIT_MouthClose,
		ARKIT_MouthDimpleLeft,
		ARKIT_MouthDimpleRight,
		ARKIT_MouthFrownLeft,
		ARKIT_MouthFrownRight,
		ARKIT_MouthFunnel,
		ARKIT_MouthLeft,
		ARKIT_MouthLowerDownLeft,
		ARKIT_MouthLowerDownRight,
		ARKIT_MouthPressLeft,
		ARKIT_MouthPressRight,
		ARKIT_MouthPucker,
		ARKIT_MouthRight,
		ARKIT_MouthRollLower,
		ARKIT_MouthRollUpper,
		ARKIT_MouthShrugLower,
		ARKIT_MouthShrugUpper,
		ARKIT_MouthSmileLeft,
		ARKIT_MouthSmileRight,
		ARKIT_MouthStretchLeft,
		ARKIT_MouthStretchRight,
		ARKIT_MouthUpperUpLeft,
		ARKIT_MouthUpperUpRight,
		ARKIT_NoseSneerLeft,
		ARKIT_NoseSneerRight,
		ARKIT_TongueOut,
	}_FaceBlendShapeARKit_;



	//Face BlendShape index
	typedef enum FACEBLENDSHAPEAUDIO
	{
		AUDIO_a = 0,
		AUDIO_b,
		AUDIO_c,
		AUDIO_d,
		AUDIO_e,
		AUDIO_f,
		AUDIO_g,
		AUDIO_h,
		AUDIO_i,
		AUDIO_j,
		AUDIO_k,
		AUDIO_l,
		AUDIO_m,
		AUDIO_n,
		AUDIO_o,
		AUDIO_p,
		AUDIO_q,
		AUDIO_r,
		AUDIO_s,
		AUDIO_t,
		AUDIO_u,
		AUDIO_v,
		AUDIO_w,
		AUDIO_x,
		AUDIO_y,
		AUDIO_z,
	}_FaceBlendShapeAudio_;



	typedef enum SENSORSTATE
	{
		SS_NONE = 0,
		SS_Well,                        //正常
		SS_NoData,                      //无数据
		SS_UnReady,                     //初始化中
		SS_BadMag,                      //磁干扰
	}_SensorState_;



	typedef enum WORLDSPACE
	{
		WS_Geo = 0,                        //表示世界坐标系为地理坐标系
		WS_Unity,                          //表示世界坐标系为Unity世界坐标系
		WS_UE4,                            //表示世界坐标系为UE4世界坐标系
	}_WorldSpace_;

	typedef enum GESTURE
	{
		GESTURE_NONE = 0,  // 未知手势
		// 1 ~ 10
		GESTURE_1,         // 指向：食指伸直，其它手指握拢
		GESTURE_2,         // 剪刀手
		GESTURE_3,         // OK
		GESTURE_4,         // 四
		GESTURE_5,         // 掌（布）
		GESTURE_6,         // 六
		GESTURE_7,         // 七
		GESTURE_8,         // 九
		GESTURE_9,         // 【暂无】
		GESTURE_10,        // 【暂无】
		// 11 ~ 20
		GESTURE_11,        // 比心
		GESTURE_12,        // 我爱你：大拇指、食指、小指伸直，其它手指握拢
		GESTURE_13,        // 摇滚：食指和尾指伸直，其它握拢
		GESTURE_14,        // 点赞
		GESTURE_15,        // 抓取
		GESTURE_16,        // 握拳（石头）
		GESTURE_17,        // 手枪：大拇指伸出，食指和中指伸直并拢
		GESTURE_18,        // 【暂无】
		GESTURE_19,        // 【暂无】
		GESTURE_20,        // 竖中指
		// 21 ~ 22
		GESTURE_21,        // 竖尾指
		GESTURE_22,        // 三
	}_Gesture_;



	//对于获取导入的数据，若 isUpdate = false 表示该帧未更新（及该帧是丢帧的补上的，且数据与前一帧相同）
	typedef struct MOCAPDATA
	{
		bool isUpdate;                              //true表示设备数据已更新
		unsigned int frameIndex;                    //帧序号
		int frequency;                              //设备数据传输频率

		int nsResult;                               //其它数据

		_SensorState_ sensorState_body[NODES_BODY] = { SS_NONE };
		float position_body[NODES_BODY][3]/*xyz-m*/ = { {0} };
		float quaternion_body[NODES_BODY][4]/*wxyz*/ = { {0} };
		float gyr_body[NODES_BODY][3] = { 0 };
		float acc_body[NODES_BODY][3] = { 0 }; //去重力加速度
		float velocity_body[NODES_BODY][3] = { 0 };

		_SensorState_ sensorState_rHand[NODES_HAND] = { SS_NONE };
		float position_rHand[NODES_HAND][3]/*xyz-m*/ = { {0} };
		float quaternion_rHand[NODES_HAND][4]/*wxyz*/ = { {0} };
		float gyr_rHand[NODES_HAND][3] = { 0 };
		float acc_rHand[NODES_HAND][3] = { 0 };  //去重力加速度
		float velocity_rHand[NODES_HAND][3] = { 0 };

		_SensorState_ sensorState_lHand[NODES_HAND] = { SS_NONE };
		float position_lHand[NODES_HAND][3]/*xyz-m*/ = { {0} };
		float quaternion_lHand[NODES_HAND][4]/*wxyz*/ = { {0} };
		float gyr_lHand[NODES_HAND][3] = { 0 };
		float acc_lHand[NODES_HAND][3] = { 0 };  //去重力加速度
		float velocity_lHand[NODES_HAND][3] = { 0 };

		bool isUseFaceBlendShapesARKit = false;
		bool isUseFaceBlendShapesAudio = false;
		float faceBlendShapesARKit[NODES_FACEBS_ARKIT] = { 0 };  //表情 bs 数据，摄像头方式识别的表情
		float faceBlendShapesAudio[NODES_FACEBS_AUDIO] = { 0 };  //表情 bs 数据，语音方式识别的表情
		float localQuat_RightEyeball[4] = { 0 };  //右眼球数据
		float localQuat_LeftEyeball[4] = { 0 };   //左眼球数据

		_Gesture_ gestureResultL = GESTURE_NONE;   //左手手势数据  
		_Gesture_ gestureResultR = GESTURE_NONE;   //右手手势数据

	}_MocapData_;


	typedef struct MOCAPDATAPLUS
	{
		bool isUpdate;                              //true表示设备数据已更新
		int id = 0;
		unsigned int frameIndex;                    //帧序号
		int frequency;                              //设备数据传输频率
		long time;
		int nsResult;                               //其它数据

		bool haveBody = false;
		bool haveRHand = false;
		bool haveLHand = false;
		bool haveFace = false;

		_SensorState_ sensorState_body[NODES_BODY] = { SS_NONE };
		float position_body[NODES_BODY][3]/*xyz-m*/ = { {0} };
		float quaternion_body[NODES_BODY][4]/*wxyz*/ = { {0} };
		float gyr_body[NODES_BODY][3] = { 0 };
		float acc_body[NODES_BODY][3] = { 0 }; //去重力加速度
		float velocity_body[NODES_BODY][3] = { 0 };

		_SensorState_ sensorState_rHand[NODES_HAND] = { SS_NONE };
		float position_rHand[NODES_HAND][3]/*xyz-m*/ = { {0} };
		float quaternion_rHand[NODES_HAND][4]/*wxyz*/ = { {0} };
		float gyr_rHand[NODES_HAND][3] = { 0 };
		float acc_rHand[NODES_HAND][3] = { 0 };  //去重力加速度
		float velocity_rHand[NODES_HAND][3] = { 0 };

		_SensorState_ sensorState_lHand[NODES_HAND] = { SS_NONE };
		float position_lHand[NODES_HAND][3]/*xyz-m*/ = { {0} };
		float quaternion_lHand[NODES_HAND][4]/*wxyz*/ = { {0} };
		float gyr_lHand[NODES_HAND][3] = { 0 };
		float acc_lHand[NODES_HAND][3] = { 0 };  //去重力加速度
		float velocity_lHand[NODES_HAND][3] = { 0 };

		bool isUseFaceBlendShapesARKit = false;
		bool isUseFaceBlendShapesAudio = false;
		float faceBlendShapesARKit[NODES_FACEBS_ARKIT] = { 0 };  //表情 bs 数据，摄像头方式识别的表情
		float faceBlendShapesAudio[NODES_FACEBS_AUDIO] = { 0 };  //表情 bs 数据，语音方式识别的表情
		float localQuat_RightEyeball[4] = { 0 };  //右眼球数据
		float localQuat_LeftEyeball[4] = { 0 };   //左眼球数据

		_Gesture_ gestureResultL = GESTURE_NONE;   //左手手势数据  
		_Gesture_ gestureResultR = GESTURE_NONE;   //右手手势数据

		int haveTracker[TRACKER_BODY] = { false };					 //光学追踪器存在标记 
		float position_Tracker[TRACKER_BODY][3] = { 0 };			//光学追踪器坐标 
		float quat_Tracker[TRACKER_BODY][4] = { 0 };				//光学追踪器姿态 
		bool haveVRglass = false;									 //VR眼镜追踪器存在标记		
		float position_VRglass[3] = { 0 };						//VR眼镜追踪器坐标
		float quat_VRglass[4] = { 0 };							//VR眼镜追踪器姿态

		bool haveTrackerR = false;									 //光学追踪器存在标记		
		float position_TrackerR[3] = { 0 };						//光学追踪器坐标
		float quat_TrackerR[4] = { 0 };							//光学追踪器姿态

		bool haveTrackerL = false;									  //光学追踪器存在标记	
		float position_TrackerL[3] = { 0 };						  //光学追踪器坐标
		float quat_TrackerL[4] = { 0 };							  //光学追踪器姿态

	}_MocapDataPlus_;



	//
	typedef struct VERSION
	{
		unsigned char Project_Name[26] = { 0 };
		unsigned char Author_Organization[128] = { 0 };
		unsigned char Author_Domain[26] = { 0 };
		unsigned char Author_Maintainer[26] = { 0 };
		unsigned char Version[26] = { 0 };
		unsigned char Version_Major;
		unsigned char Version_Minor;
		unsigned char Version_Patch;
	}_Version_;



	typedef enum G1NODES
	{
		left_hip_pitch_joint,
		left_hip_roll_joint,
		left_hip_yaw_joint,
		left_knee_joint,
		left_ankle_pitch_joint,
		left_ankle_roll_joint,
		right_hip_pitch_joint,
		right_hip_roll_joint,
		right_hip_yaw_joint,
		right_knee_joint,
		right_ankle_pitch_joint,
		right_ankle_roll_joint,
		waist_yaw_joint,
		waist_roll_joint,			// 23 自由度版本没有该关节
		waist_pitch_joint,			// 23 自由度版本没有该关节
		left_shoulder_pitch_joint,
		left_shoulder_roll_joint,
		left_shoulder_yaw_joint,
		left_elbow_joint,
		left_wrist_roll_joint,
		left_wrist_pitch_joint,		// 23 自由度版本没有该关节
		left_wrist_yaw_joint,		// 23 自由度版本没有该关节
		right_shoulder_pitch_joint,
		right_shoulder_roll_joint,
		right_shoulder_yaw_joint,
		right_elbow_joint,
		right_wrist_roll_joint,
		right_wrist_pitch_joint,	// 23 自由度版本没有该关节
		right_wrist_yaw_joint,		// 23 自由度版本没有该关节
	}_G1Nodes;


	typedef enum DEX31NODES
	{
		left_hand_thumb_0_joint,
		left_hand_thumb_1_joint,
		left_hand_thumb_2_joint,
		left_hand_index_0_joint,
		left_hand_index_1_joint,
		left_hand_middle_0_joint,
		left_hand_middle_1_joint,
	}_Dex31Nodes_;


	typedef enum INSPIREHANDNODES
	{
		L_thumb_proximal_yaw_joint,		// left_thumb_1_joint
		L_thumb_proximal_pitch_joint,	// left_thumb_2_joint
		L_thumb_intermediate_joint,		// left_thumb_3_joint
		L_thumb_distal_joint,			// left_thumb_4_joint
		L_index_proximal_joint,			// left_index_1_joint
		L_index_intermediate_joint,		// left_index_2_joint
		L_middle_proximal_joint,		// left_middle_1_joint
		L_middle_intermediate_joint,	// left_middle_2_joint
		L_ring_proximal_joint,			// left_ring_1_joint
		L_ring_intermediate_joint,		// left_ring_2_joint
		L_pinky_proximal_joint,			// left_little_1_joint
		L_pinky_intermediate_joint,		// left_little_2_joint
	}_InspireHandNodes_;

}//end namespace

#endif // !VDMOCAPSDK_DATAREAD_DATATYPE_H

