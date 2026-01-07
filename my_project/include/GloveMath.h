#pragma once
#include "VDGloveDataTypes.h"
#include <iostream>

class GloveMath {
public:
    // 常量定义：VD 到 MANO 的旋转矩阵
    static Eigen::Matrix3f get_VD_Right_To_Mano() {
        Eigen::Matrix3f m;
        m << -1, 0, 0,
              0, 0, 1,
              0, 1, 0;
        return m;
    }

    static Eigen::Matrix3f get_VD_Left_To_Mano() {
        Eigen::Matrix3f m;
        m << -1, 0, 0,
              0, 0, 1,
              0, 1, 0;
        return m;
    }

    // Python: quaternion_average
    static Eigen::Vector4f quaternion_average(const Eigen::Vector4f& q1, const Eigen::Vector4f& q2) {
        Eigen::Vector4f q2_adj = q2;
        if (q1.dot(q2) < 0) {
            q2_adj = -q2;
        }
        Eigen::Vector4f q_sum = q1 + q2_adj;
        return q_sum.normalized();
    }

    // 辅助：计算两个向量之间的旋转 (对应 scipy 的处理)
    static Eigen::Quaternionf get_rotation_between_vecs(const Eigen::Vector3f& v_from, const Eigen::Vector3f& v_to) {
        Eigen::Vector3f v_from_u = v_from.normalized();
        Eigen::Vector3f v_to_u = v_to.normalized();
        return Eigen::Quaternionf::FromTwoVectors(v_from_u, v_to_u);
    }

    // Python: add_finger_tip_point
    static Eigen::Vector3f add_finger_tip_point(
        const Eigen::Vector3f& A, const Eigen::Vector3f& B, 
        const Eigen::Vector3f& C, const Eigen::Vector3f& D) 
    {
        Eigen::Vector3f v_AB = B - A;
        Eigen::Vector3f v_BC = C - B;
        Eigen::Vector3f v_CD = D - C;

        float len_BC = v_BC.norm();
        float len_CD = v_CD.norm();
        float len_DE = (len_BC + len_CD) / 2.0f * 0.8f;

        // 计算旋转 R1 (BC -> CD) 和 R2 (AB -> BC)
        Eigen::Quaternionf R1 = get_rotation_between_vecs(v_BC, v_CD);
        Eigen::Quaternionf R2 = get_rotation_between_vecs(v_AB, v_BC);

        // Python: R_delta = R2.inv() * R1
        // Python: R_mean = R2 * (R_delta ** 0.5) -> Slerp
        Eigen::Quaternionf R_delta = R2.conjugate() * R1; 
        
        // Slerp 一半相当于 delta ** 0.5 (从 Identity 插值到 R_delta)
        Eigen::Quaternionf R_delta_half = Eigen::Quaternionf::Identity().slerp(0.5f, R_delta);
        
        Eigen::Quaternionf R_mean = R2 * R_delta_half;

        Eigen::Vector3f v_CD_norm = v_CD.normalized();
        Eigen::Vector3f v_DE_dir = R_mean._transformVector(v_CD_norm);

        return D + (v_DE_dir * len_DE);
    }

    // Python: VD_to_mano_keypoints
    // 注意：这里输入的是原始 keypoints (21个点，其中指尖可能是未计算的或者错的，根据逻辑需要重算)
    // hand_wrist_rot 是 Matrix3f
    static std::vector<Eigen::Vector3f> process_hand_to_mano(
        const std::vector<Eigen::Vector3f>& raw_pos, // 原始 20/21 点
        const Eigen::Matrix3f& wrist_rot_mat,
        bool is_right_hand) 
    {
        // 1. 复制数据，准备处理
        std::vector<Eigen::Vector3f> processed = raw_pos;
        
        // 确保数据够用，如果原始只有20个点，需要扩容到21
        if (processed.size() < 21) processed.resize(21);

        // 2. 将手腕点 (Index 0) 移到原点
        Eigen::Vector3f wrist_pos = processed[0];
        for (auto& p : processed) {
            p -= wrist_pos;
        }

        // 3. 定义索引 (Python: finger_starts = [5, 9, 13, 17])
        // 对应 食指、中指、无名指、小指
        std::vector<int> finger_starts = {5, 9, 13, 17};

        for (int start_idx : finger_starts) {
            int idx_B = start_idx;     // MCP
            int idx_C = start_idx + 1; // PIP
            int idx_D = start_idx + 2; // DIP
            int idx_Tip = start_idx + 3; // Tip (Target)

            // Python Logic: A_in 来自 hand_keypoints[..., 0, :] 即手腕
            Eigen::Vector3f A = processed[0]; 
            Eigen::Vector3f B = processed[idx_B];
            Eigen::Vector3f C = processed[idx_C];
            Eigen::Vector3f D = processed[idx_D];

            // 计算新的指尖位置
            processed[idx_Tip] = add_finger_tip_point(A, B, C, D);
        }

        // 4. 旋转调整: rotated_kp = np.linalg.inv(hand_wrist_rot) @ kp
        Eigen::Matrix3f rot_inv = wrist_rot_mat.inverse();
        for (auto& p : processed) {
            p = rot_inv * p;
        }

        // 5. 转换到 MANO 坐标系
        Eigen::Matrix3f trans_mat = is_right_hand ? get_VD_Right_To_Mano() : get_VD_Left_To_Mano();
        
        // Python: keypoints @ Matrix (相当于 p^T * M)
        // Eigen: M^T * p (如果是列向量) 
        // 验证 Python: (N, 3) @ (3, 3) -> (N, 3). 每一行 vec * Mat.
        // 在 Eigen 中通常处理列向量 v. 如果 Python 是 v_row * M, 则 Eigen 是 M.transpose() * v_col
        for (auto& p : processed) {
            p = trans_mat.transpose() * p;
        }

        return processed;
    }
};