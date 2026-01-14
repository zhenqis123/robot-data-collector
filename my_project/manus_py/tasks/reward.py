import torch

def reward_resdex(
    reset_buf,
    progress_buf,
    successes,
    current_successes,
    has_hit_table,
    max_episode_length: float,
    table_heights,
    object_pos,
    palm_pos,
    fingertip_pos,
    num_fingers: int,
    actions,
    object_init_states,
    distractor_object_pos=None,
    **kwargs,
):
    info = {}
    object_delta_z = object_pos[:, 2] - object_init_states[:, 2]
    palm_object_dist = torch.norm(object_pos - palm_pos, dim=-1)
    palm_object_dist = torch.where(palm_object_dist >= 0.5, 0.5, palm_object_dist)
    horizontal_offset = torch.norm(object_pos[:, 0:2], dim=-1) #- object_init_states[:, 0:2], dim=-1)

    fingertips_object_dist = torch.zeros_like(object_delta_z)
    for i in range(fingertip_pos.shape[-2]):
        fingertips_object_dist += torch.norm(fingertip_pos[:, i, :] - object_pos, dim=-1)
    fingertips_object_dist = torch.where(fingertips_object_dist >= 3.0, 3.0, fingertips_object_dist)

    flag = torch.logical_or((fingertips_object_dist <= 0.12 * num_fingers), (palm_object_dist <= 0.15))

    # after hand approach object, lift_object
    lift_object = torch.zeros_like(object_delta_z)
    lift_object = torch.where(flag, object_delta_z, lift_object)

    reward = (
        -0.5 * fingertips_object_dist
        - palm_object_dist
        + 3.0 * lift_object
        #+ bonus
        #- 0.3 * horizontal_offset
    )

    info["fingertips_object_dist"] = fingertips_object_dist
    info["palm_object_dist"] = palm_object_dist
    info["lift_object"] = lift_object
    info["horizontal_offset"] = horizontal_offset
    info["reward"] = reward
    info["hand_approach_flag"] = flag

    resets = reset_buf.clone()
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)
    progress_buf = torch.where(resets > 0, torch.zeros_like(progress_buf), progress_buf)
    #resets = torch.where(object_pos[:, 2] <= 0.3, torch.ones_like(resets), resets)
    
    successes = torch.where(
        object_delta_z > 0.1,
        torch.where(
            flag,
            torch.ones_like(successes),
            torch.zeros_like(successes),
        ),
        torch.zeros_like(successes),
    )

    # # for any-demograsp, grasping any object is considered success
    # if distractor_object_pos is not None:
    #     distractor_z = distractor_object_pos[:, 2].reshape(successes.shape[0], -1) # [num_envs, num_distractors]
    #     distractor_z = distractor_z.max(dim=-1).values
    #     successes = torch.where(
    #         distractor_z > 0.1,
    #         torch.ones_like(successes),
    #         successes,
    #     )


    #num_resets = torch.sum(resets)
    #finished_cons_successes = torch.sum(successes * resets.float())
    current_successes = torch.where(resets>0, successes, current_successes)
    #print(current_successes, successes, resets)

    # check robot-table collision
    min_keypoint_z = torch.min(fingertip_pos[:, :, 2], dim=-1).values
    min_keypoint_z = torch.min(min_keypoint_z, palm_pos[:, 2])
    has_hit_table = torch.where(
        min_keypoint_z < table_heights,
        torch.ones_like(has_hit_table, dtype=torch.bool),
        has_hit_table
    )

    #print(has_hit_table[8], min_keypoint_z[8], table_heights[8])

    #print("resdex reward:", reward)
    return (
        reward,
        resets,
        progress_buf,
        successes,
        current_successes,
        has_hit_table,
        info,
    )


# def reward_track(
#     reset_buf,
#     progress_buf,
#     successes,
#     current_successes,
#     max_episode_length: float,
#     object_pos,
#     palm_pos,
#     fingertip_pos,
#     num_fingers: int,
#     actions,
#     object_init_states,
#     tracking_timestep,
#     end_effector_pose,
#     hand_qpos,
#     tracking_reference,
#     **kwargs,
# ):
#     info = {}
    
#     all_batch = torch.arange(object_pos.shape[0], device=object_pos.device)
#     ref_wrist_initobj_pos = tracking_reference["wrist_initobj_pos"][all_batch, tracking_timestep]
#     ref_wrist_quat = tracking_reference["wrist_quat"][all_batch, tracking_timestep]
#     ref_hand_qpos = tracking_reference["hand_qpos"][all_batch, tracking_timestep]
#     ref_obj_initobj_pos = tracking_reference["obj_initobj_pos"][all_batch, tracking_timestep]
    
#     cur_wrist_initobj_pos = end_effector_pose[:, 0:3] - object_init_states[:, 0:3]
#     cur_wrist_quat = end_effector_pose[:, 3:7]
#     cur_hand_qpos = hand_qpos
#     cur_obj_initobj_pos = object_pos[:, 0:3] - object_init_states[:, 0:3]

#     ## compute distances to reference: use euclidean distance for positions, use angles between two quats for orientations,
#     # use abs difference for joint qpos
#     dist_wrist_initobj_pos = torch.norm(cur_wrist_initobj_pos - ref_wrist_initobj_pos, dim=-1)
#     dot = (cur_wrist_quat * ref_wrist_quat).sum(dim=-1)  # 计算点积
#     dist_wrist_quat = 2 * torch.acos(torch.clamp(torch.abs(dot), 0, 1))
#     dist_hand_qpos = torch.abs(cur_hand_qpos - ref_hand_qpos).mean(dim=-1) # TODO: we have changed handqpos to hand target qpos in the reference
#     dist_obj_initobj_pos = torch.norm(cur_obj_initobj_pos - ref_obj_initobj_pos, dim=-1)

#     # 调整reward系数，在误差上让0.01m、0.25手腕旋转、和0.5关节角对reward都产生约0.1的影响
#     reward = 0.1 * (
#         - dist_wrist_initobj_pos / 0.01
#         - dist_wrist_quat / 0.25
#         - dist_hand_qpos / 0.5
#         - dist_obj_initobj_pos / 0.01
#     )

#     # For tracking_timestep>0 envs, go to track next step; but do not exceed T_ref-1
#     tracking_timestep[:] = torch.where(
#         tracking_timestep > 0,
#         tracking_timestep + 1,
#         tracking_timestep
#     )
#     T_ref = tracking_reference["wrist_initobj_pos"].shape[1]
#     tracking_timestep[:] = torch.min(tracking_timestep, torch.ones_like(tracking_timestep) * (T_ref - 1))

#     # For tracking_timestep==0 envs, if close enough to the reference, go to step 1
#     flag_stage0_done = (dist_wrist_initobj_pos < 0.02) & \
#         (dist_wrist_quat < 0.25) & \
#         (dist_hand_qpos < 0.5) & \
#         (dist_obj_initobj_pos < 0.01) & \
#         (tracking_timestep == 0)
#     tracking_timestep[:] = torch.where(
#         flag_stage0_done,
#         torch.ones_like(tracking_timestep) * 1,
#         tracking_timestep
#     )

#     info["dist_wrist_initobj_pos"] = dist_wrist_initobj_pos
#     info["dist_wrist_quat"] = dist_wrist_quat
#     info["dist_hand_qpos"] = dist_hand_qpos
#     info["dist_obj_initobj_pos"] = dist_obj_initobj_pos
#     info["reward"] = reward

#     # print("Track:", tracking_timestep.cpu().numpy(), 
#     #       "Dis:", dist_wrist_initobj_pos.cpu().numpy(), dist_wrist_quat.cpu().numpy(), 
#     #       dist_hand_qpos.cpu().numpy(), dist_obj_initobj_pos.cpu().numpy())
#     # print(reward.cpu().numpy())

#     resets = reset_buf.clone()
#     resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)
#     progress_buf = torch.where(resets > 0, torch.zeros_like(progress_buf), progress_buf)
#     #resets = torch.where(object_pos[:, 2] <= 0.3, torch.ones_like(resets), resets)
    
#     object_delta_z = object_pos[:, 2] - object_init_states[:, 2]
#     successes = torch.where(
#         object_delta_z > 0.2,
#         torch.ones_like(successes),
#         torch.zeros_like(successes),
#     )
#     #num_resets = torch.sum(resets)
#     #finished_cons_successes = torch.sum(successes * resets.float())
#     current_successes = torch.where(resets>0, successes, current_successes)

#     return (
#         reward,
#         resets,
#         progress_buf,
#         successes,
#         current_successes,
#         info,
#     )


REWARD_DICT = {
    "resdex": reward_resdex, 
    #"track": reward_track,
}