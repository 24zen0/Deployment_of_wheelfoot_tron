import os
import sys
from typing import Dict

import torch
from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi
from legged_gym.utils.terrain import Terrain
from collections import deque
from scipy.stats import vonmises
import pandas as pd
from tqdm import tqdm

class PointFootEE:
    
    def system_id(self, motor_data_file):
        # load motor_data_file, csv
        motor_data = pd.read_csv(motor_data_file)
        motor_q = motor_data[["q_real0", "q_real1", "q_real2",
                              "q_real3", "q_real4", "q_real5"]].to_numpy()
        # motor_dq = motor_data[["jvel0","jvel1","jvel2","jvel3","jvel4","jvel5"]].to_numpy()
        motor_q_des = motor_data[["q_des0", "q_des1", "q_des2",
                                  "q_des3", "q_des4", "q_des5"]].to_numpy()

        q_real = torch.from_numpy(motor_q).float().to(self.device)
        qd_real = torch.zeros_like(q_real).to(self.device)
        q_des = torch.from_numpy(motor_q_des).float().to(self.device)

        # sample parameters
        joint_damping_range = self.cfg.sysid_param_range.joint_damping_range
        joint_friction_range = self.cfg.sysid_param_range.joint_friction_range
        joint_armature_range = self.cfg.sysid_param_range.joint_armature_range
        # set parameters
        sampled_dampings = np.zeros((self.num_envs,))
        sampled_frictions = np.zeros((self.num_envs,))
        sampled_armatures = np.zeros((self.num_envs,))
        for i in range(self.num_envs):
            # set joint_friction and joint_damping
            dof_props = self.gym.get_actor_dof_properties(
                self.envs[i], self.actor_handles[i])
            joint_damping = np.random.uniform(
                joint_damping_range[0], joint_damping_range[1])
            joint_friction = np.random.uniform(
                joint_friction_range[0], joint_friction_range[1])
            joint_armature = np.random.uniform(
                joint_armature_range[0], joint_armature_range[1])
            # assume all joints have the same damping and friction
            for j in range(len(dof_props)):
                # if i == 0:
                #     joint_damping = 0.05
                #     joint_friction = 0.2
                #     joint_armature = 0.0
                sampled_dampings[i] = joint_damping
                sampled_frictions[i] = joint_friction
                sampled_armatures[i] = joint_armature
                dof_props["damping"][j] = joint_damping
                dof_props["friction"][j] = joint_friction
                dof_props["armature"][j] = joint_armature
            self.gym.set_actor_dof_properties(
                self.envs[i], self.actor_handles[i], dof_props)

        # check
        for i in range(self.num_envs):
            dof_props = self.gym.get_actor_dof_properties(
                self.envs[i], self.actor_handles[i])
            for j in range(len(dof_props)):
                assert abs(dof_props["damping"][j] -
                           sampled_dampings[i]) < 1e-4
                assert abs(dof_props["friction"][j] -
                           sampled_frictions[i]) < 1e-4
                assert abs(dof_props["armature"][j] -
                           sampled_armatures[i]) < 1e-4
        # generating samples
        metric = 0
        sim_q = []
        # reset
        env_ids = torch.arange(self.num_envs).to(self.device)
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]

        # base velocities
        self.root_states[env_ids,
                         7:13] = 0.  # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(
                                                         self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.dof_pos[env_ids] = torch.tile(q_real[0], (self.num_envs, 1))
        self.dof_vel[env_ids] = torch.tile(qd_real[0], (self.num_envs, 1))

        # Important! should feed actor id, not env id
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(
                                                  self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        delay_steps = 0
        print(f"sim_dt: {self.cfg.sim.dt}")

        for i in tqdm(range(q_real.shape[0]-1-delay_steps)):
            # apply action
            actions = ((q_des[i] - self.default_dof_pos[0]) /
                       self.cfg.control.action_scale).tile((self.num_envs, 1))
            # step physics and render each frame
            self.render()
            self.torques = self._compute_torques(
                actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.foot_positions = self.rigid_body_states[:,
                                                         self.feet_indices, 0:3]
            # when sampling
            metric = metric + \
                torch.norm(self.dof_pos -
                           q_real[i+1+delay_steps].unsqueeze(dim=0), dim=-1)
            sim_q.append(self.dof_pos.cpu().numpy())

        metric = metric.detach().cpu().numpy()
        all_coms = []
        all_masses = []
        all_restitution = []
        for i in range(len(self.envs)):
            com = self.gym.get_actor_rigid_body_properties(
                self.envs[i], self.actor_handles[i])[0].com
            masses = [self.gym.get_actor_rigid_body_properties(
                self.envs[i], self.actor_handles[i])[j].mass for j in range(self.num_bodies)]
            all_coms.append(np.array([com.x, com.y, com.z]))
            all_masses.append(np.array(masses))
            all_restitution.append(self.gym.get_actor_rigid_shape_properties(
                self.envs[i], self.actor_handles[i])[0].restitution)
        all_coms = np.array(all_coms)
        all_masses = np.array(all_masses)
        all_restitution = np.array(all_restitution)
        # print("Average metric", np.mean(metric))
        print("best")
        print("damping", sampled_dampings[np.argmin(metric)], "\n",
              "friction", sampled_frictions[np.argmin(metric)], "\n",
              "armature", sampled_armatures[np.argmin(metric)], "\n",
              #   "feet_friction", self.gym.get_actor_rigid_shape_properties(self.envs[np.argmin(metric)], self.actor_handles[np.argmin(metric)])[self.feet_indices[0]].friction,
              #   "rb_restitution", self.gym.get_actor_rigid_shape_properties(self.envs[np.argmin(metric)], self.actor_handles[np.argmin(metric)])[0].restitution,
              #   "mass", [self.gym.get_actor_rigid_body_properties(self.envs[np.argmin(metric)], self.actor_handles[np.argmin(metric)])[i].mass for i in range(self.num_bodies)],
              #   "com", self.gym.get_actor_rigid_body_properties(self.envs[np.argmin(metric)], self.actor_handles[np.argmin(metric)])[0].com,
              "metric", metric[np.argmin(metric)])
        # print("worst", "damping", sampled_dampings[np.argmax(metric)],
        #       "friction", sampled_frictions[np.argmax(metric)],
        #       "limb_mass_ratios", self.sampled_limb_mass_scales[np.argmax(metric)],
        #     #   "feet_friction", self.gym.get_actor_rigid_shape_properties(self.envs[np.argmax(metric)], self.actor_handles[np.argmax(metric)])[self.feet_indices[0]].friction,
        #     #   "rb_restitution", self.gym.get_actor_rigid_shape_properties(self.envs[np.argmax(metric)], self.actor_handles[np.argmax(metric)])[0].restitution,
        #     #   "mass", [self.gym.get_actor_rigid_body_properties(self.envs[np.argmax(metric)], self.actor_handles[np.argmax(metric)])[i].mass for i in range(self.num_bodies)],
        #     #   "com", self.gym.get_actor_rigid_body_properties(self.envs[np.argmax(metric)], self.actor_handles[np.argmax(metric)])[0].com,
        #        "kp", kp_des[np.argmax(metric)],
        #        "kd", kd_des[np.argmax(metric)],
        #         "armature", sampled_armatures[np.argmax(metric)],
        #        metric[np.argmax(metric)])
    
    def generate_qdes_data(self, file_path):
        """
        generate q_des data for real robot data collection
        """
        assert self.num_envs == 1, "Only one env is supported for IK validation"
        frequency = int(1 / (self.cfg.sim.dt * self.cfg.control.decimation))
        rollout_length = frequency * 60  # frequency for 60s
        data = np.ndarray((rollout_length, 6), dtype=np.float32)

        self._reset_root_states(torch.tensor([0], device=self.device))
        self._reset_dofs(torch.tensor([0], device=self.device))
        q_des = torch.zeros_like(self.default_dof_pos.squeeze(0))
        [hip_max, hip_min] = self.cfg.joint_pos_des_range.hip_joint
        [thigh_max, thigh_min] = self.cfg.joint_pos_des_range.thigh_joint
        [calf_max, calf_min] = self.cfg.joint_pos_des_range.calf_joint
        
        hip_average = (hip_max + hip_min) / 2
        thigh_average = (thigh_max + thigh_min) / 2
        calf_average = (calf_max + calf_min) / 2

        # generate sin waves of different frequencies, each lasts for 10 seconds
        t = 0.0
        for i in range(rollout_length):
            sin_frequency = (i // (10 * frequency)) * 0.3 + 0.2  # frequency increases every 10 seconds
            sin_wave = np.sin(2 * np.pi * t * sin_frequency)
            hip_sin = hip_average + (hip_max - hip_average) * sin_wave
            thigh_sin = thigh_average + (thigh_max - thigh_average) * sin_wave
            calf_sin = calf_average + (calf_max - calf_average) * sin_wave
            q_des[0] = hip_sin
            q_des[1] = thigh_sin
            q_des[2] = calf_sin
            q_des[3] = hip_sin
            q_des[4] = thigh_sin
            q_des[5] = calf_sin

            print(f"sin_frequency: {sin_frequency}")
            data[i] = q_des.cpu().numpy()
            # visualize for debugging
            actions = ((q_des - self.default_dof_pos) / self.cfg.control.action_scale)
            # step physics and render each frame
            self.render()
            for _ in range(self.cfg.control.decimation):
                self.torques = self._compute_torques(actions).view(self.torques.shape)
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
                self.gym.simulate(self.sim)
                if self.device == 'cpu':
                    self.gym.fetch_results(self.sim, True)
                self.gym.refresh_dof_state_tensor(self.sim)
            
            t += self.dt
            if t > 10.0:
                t = 0.0

        df = pd.DataFrame(data, columns=[
                          "q_des_0", "q_des_1", "q_des_2", "q_des_3", "q_des_4", "q_des_5"])
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
        print("sample jpos_des done")
    
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = self.cfg.env.debug_viz
        self.init_done = False
        self._parse_cfg()
        self.gym = gymapi.acquire_gym()

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(
            self.sim_device)
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type == 'cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id

        self.num_envs = cfg.env.num_envs
        self.num_estimator_features = cfg.env.num_estimator_features
        self.num_estimator_labels = cfg.env.num_estimator_labels
        self.num_obs = self.num_estimator_features + self.num_estimator_labels
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.estimator_features_buf = torch.zeros(
            self.num_envs, self.num_estimator_features, device=self.device, dtype=torch.float)
        self.estimator_labels_buf = torch.zeros(
            self.num_envs, self.num_estimator_labels, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device,
                                                  dtype=torch.float)
        else:
            self.privileged_obs_buf = None

        self.extras = {}

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def get_observations(self):
        return self.estimator_features_buf, self.estimator_labels_buf, self.privileged_obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        estimator_features_buf, estimator_labels_buf, privileged_obs, _, _, _ = self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return estimator_features_buf, estimator_labels_buf, privileged_obs

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(
            actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(
                self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.estimator_features_buf = torch.clip(
            self.estimator_features_buf, -clip_obs, clip_obs)
        self.estimator_labels_buf = torch.clip(
            self.estimator_labels_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.estimator_features_buf, self.estimator_labels_buf, self.privileged_obs_buf, \
            self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights_actor or self.cfg.terrain.measure_heights_critic:
            self.measured_heights = self._get_heights()
            self._calc_point_heights_around_feet()
        self._refresh_rigid_body_states()

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()

        # Periodic Reward Framework phi cycle
        # step after computing reward but before resetting the env
        self.gait_time += self.dt
        # +self.dt/2 in case of float precision errors
        is_over_limit = (self.gait_time >= (self.gait_period - self.dt / 2))
        over_limit_indices = is_over_limit.nonzero(as_tuple=False).flatten()
        self.gait_time[over_limit_indices] = 0.0
        self.phi = self.gait_time / self.gait_period

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        # in some cases a simulation step might be required to refresh some obs (for example body positions)
        self._calc_periodic_reward_obs()
        self.compute_observations()

        self.llast_actions[:] = self.actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def _calc_periodic_reward_obs(self):
        """Calculate the periodic reward observations.
        """
        for i in range(2):
            self.clock_input[:, i] = torch.sin(
                2 * torch.pi * (self.phi + self.theta[:, i].unsqueeze(1))).squeeze(-1)

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
                                   dim=1)
        # no terminal reward for time-outs
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        self._episodic_domain_randomization(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)
        self._resample_gait_params(env_ids)

        self._reset_buffers(env_ids)
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(
                self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _episodic_domain_randomization(self, env_ids):
        """ Update scale of Kp, Kd, rfi lim"""
        if len(env_ids) == 0:
            return

        if self.cfg.domain_rand.randomize_pd_gain:

            self._kp_scale[env_ids] = torch_rand_float(
                self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), self.num_actions), device=self.device)
            self._kd_scale[env_ids] = torch_rand_float(
                self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), self.num_actions), device=self.device)

    def _reset_buffers(self, env_ids):
        # reset buffers
        self.llast_actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # Periodic Reward Framework
        self.gait_time[env_ids] = 0.0
        self.phi[env_ids] = 0.0
        self.clock_input[env_ids, :] = 0.0
        # clear obs and critic history for the envs that are reset
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination(
            ) * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        """ Computes observations
        """

        obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            self.clock_input,
            self.base_height_target,
        ), dim=-1)

        if self.cfg.terrain.measure_heights_actor:
            obs_buf = self._add_height_measure_to_buf(
                obs_buf)
        
        # compute estimator labels
        # feet_relative heights([num_envs, 8]), 4 points around each foot
        # [left_foot_back, left_foot_front, left_foot_right, left_foot_left, 
        # right_foot_back, right_foot_front, right_foot_right, right_foot_left]
        self.feet_relative_heights[:, 0:4] = self.foot_pos[:, 0, 2].unsqueeze(1).repeat(1, 4) # left foot heights, repeated 4 times
        self.feet_relative_heights[:, 4:8] = self.foot_pos[:, 1, 2].unsqueeze(1).repeat(1, 4) # right foot heights, repeated 4 times
        self.feet_relative_heights -= self.height_around_feet.flatten(1, 2)  # subtract the height around feet

        self.estimator_labels_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            (torch.abs(self.contact_forces[:, self.feet_indices, 2])/1.0).clip(
            min=0., max=1.),                              # 2
            self.feet_relative_heights,                   # 8
        ), dim=-1)

        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.cat((
                self.commands[:, :3] * self.commands_scale,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                self.clock_input,
                self.base_height_target,
                # robot state terms
                self.base_lin_vel * self.obs_scales.lin_vel,         # 3
                self.exp_C_frc_left,                                 # 1
                self.exp_C_frc_right,                                # 1
                self.exp_C_spd_left,                                 # 1
                self.exp_C_spd_right,                                # 1
                # domain randomization terms
                self._rand_push_vels[:, :2],                         # 2
                self._inertia_scale,                                 # 9
                self._base_com_bias,                                 # 3
                self._ground_friction_values,                        # 1
                # self._restitution_values,                            # 1
                self._base_mass / 30.0,                              # 1
                self._kp_scale,                                      # 6
                self._kd_scale,                                      # 6
                self.height_around_feet.flatten(1, 2),               # 8
                self._joint_friction.unsqueeze(1),    # 1
                self._joint_damping.unsqueeze(1),     # 1
                self._joint_armature.unsqueeze(1),    # 1
            ), dim=-1)

        # add noise if needed
        if self.add_noise:
            obs_now = obs_buf.clone()
            obs_now += (2 * torch.rand_like(obs_now) - 1) * \
                self.noise_scale_vec
        else:
            obs_now = obs_buf.clone()

        self.obs_history.append(obs_now)
        self.estimator_features_buf = torch.cat(
            [self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=-1
        )
        self.critic_history.append(self.privileged_obs_buf)
        self.privileged_obs_buf = torch.cat(
            [self.critic_history[i] for i in range(self.critic_history.maxlen)], dim=-1
        )

    def _add_height_measure_to_buf(self, buf):
        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1,
                             1.) * self.obs_scales.height_measurements
        buf = torch.cat(
            (buf, heights), dim=-1
        )
        return buf

    def create_sim(self):
        """ Creates simulation, terrain and environments
        You can modify this function to add your own terrain and environment settings.
        For example, you can add a custom terrain mixed with heightfield and trimesh.
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine,
                                       self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ------------- Callbacks --------------

    def _init_domain_params(self):
        # init params for domain randomization
        # init 0 for values
        # init 1 for scales
        self._rand_push_vels = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self._base_com_bias = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self._base_mass = torch.ones(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self._ground_friction_values = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self._restitution_values = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self._inertia_scale = torch.ones(
            self.num_envs, self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)
        self._kp_scale = torch.ones(
            self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self._kd_scale = torch.ones(
            self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self._joint_friction = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self._joint_damping = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self._joint_armature = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1),
                                                    device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
            self._ground_friction_values[env_id] += self.friction_coeffs[env_id].squeeze()
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                              requires_grad=False)
            self.dof_vel_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * \
                    r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * \
                    r * self.cfg.rewards.soft_dof_pos_limit
        
        if self.cfg.domain_rand.randomize_joint_friction:
            joint_friction_range = np.array(
                self.cfg.domain_rand.joint_friction_range, dtype=np.float32)
            friction = np.random.uniform(
                joint_friction_range[0], joint_friction_range[1])
            self._joint_friction[env_id] = friction
            for j in range(self.num_dof):
                props["friction"][j] = torch.tensor(
                    friction, dtype=torch.float, device=self.device)

        if self.cfg.domain_rand.randomize_joint_damping:
            joint_damping_range = np.array(
                self.cfg.domain_rand.joint_damping_range, dtype=np.float32)
            damping = np.random.uniform(
                joint_damping_range[0], joint_damping_range[1])
            self._joint_damping[env_id] = damping
            for j in range(self.num_dof):
                props["damping"][j] = torch.tensor(
                    damping, dtype=torch.float, device=self.device)

        if self.cfg.domain_rand.randomize_joint_armature:
            joint_armature_range = np.array(
                self.cfg.domain_rand.joint_armature_range, dtype=np.float32)
            armature = np.random.uniform(
                joint_armature_range[0], joint_armature_range[1])
            self._joint_armature[env_id] = armature
            for j in range(self.num_dof):
                props["armature"][j] = torch.tensor(
                    armature, dtype=torch.float, device=self.device)
        
        return props

    def _process_rigid_body_props(self, props, env_id):
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
            self._base_mass[env_id] = props[0].mass
        if self.cfg.domain_rand.randomize_base_com:
            com_x, com_y, com_z = self.cfg.domain_rand.rand_com_vec
            com_x_bias = np.random.uniform(-com_x, com_x)
            com_y_bias = np.random.uniform(-com_y, com_y)
            com_z_bias = np.random.uniform(-com_z, com_z)
            props[0].com.x += com_x_bias
            props[0].com.y += com_y_bias
            props[0].com.z += com_z_bias
            self._base_com_bias[env_id, 0] += com_x_bias
            self._base_com_bias[env_id, 1] += com_y_bias
            self._base_com_bias[env_id, 2] += com_z_bias
        # randomize inertia
        if self.cfg.domain_rand.randomize_inertia:
            for i in range(len(props)):
                low_bound, high_bound = self.cfg.domain_rand.randomize_inertia_range
                inertia_scale = np.random.uniform(low_bound, high_bound)
                self._inertia_scale[env_id, i] *= inertia_scale
                props[i].mass *= inertia_scale
                props[i].inertia.x.x *= inertia_scale
                props[i].inertia.y.y *= inertia_scale
                props[i].inertia.z.z *= inertia_scale
        return props

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(
            as_tuple=False).flatten()
        self._resample_commands(env_ids)
        env_ids = (self.episode_length_buf % int(
            self.cfg.rewards.periodic_reward_framework.resampling_time / self.dt) == 0).nonzero(
            as_tuple=False).flatten()
        self._resample_gait_params(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0],
                                                     self.command_ranges["lin_vel_x"][1], (len(
                                                         env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0],
                                                     self.command_ranges["lin_vel_y"][1], (len(
                                                         env_ids), 1),
                                                     device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0],
                                                         self.command_ranges["heading"][1], (len(
                                                             env_ids), 1),
                                                         device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0],
                                                         self.command_ranges["ang_vel_yaw"][1], (len(
                                                             env_ids), 1),
                                                         device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(
            self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _resample_gait_params(self, env_ids):
        num_gaits = self.cfg.rewards.periodic_reward_framework.num_gaits
        if num_gaits == 1:
            # no need to resample
            return
        else:
            gait_index = torch.randint(0, num_gaits, (1,), device=self.device)
            self.gait_period[env_ids, :] = self.cfg.rewards.periodic_reward_framework.gait_period[gait_index]
            self.b_swing[env_ids, :] = self.cfg.rewards.periodic_reward_framework.b_swing[gait_index] * 2 * torch.pi
            self.theta[env_ids, 0] = self.cfg.rewards.periodic_reward_framework.theta_left[gait_index]
            self.theta[env_ids, 1] = self.cfg.rewards.periodic_reward_framework.theta_right[gait_index]
            

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = self._kp_scale * self.p_gains * (
                actions_scaled + self.default_dof_pos - self.dof_pos) - self._kd_scale * self.d_gains * self.dof_vel
        elif control_type == "V":
            torques = self._kp_scale * self.p_gains * (actions_scaled - self.dof_vel) - self._kd_scale * self.d_gains * (
                self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environment ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(-0.3, 0.3, (len(env_ids), self.num_dof),
                                                                        device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(
                                                  self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environment ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2),
                                                              device=self.device)  # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6),
                                                           device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(
                                                         self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """Random pushes the robots."""
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self._rand_push_vels[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)
        # set random base velocity in xy plane
        self.root_states[:, 7:9] += self._rand_push_vels[:, :2]
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(
            self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2],
                                           dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids] >= self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids],
                                                                      self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids],
                                                              0))  # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids],
                                                         self.terrain_types[env_ids]]

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * \
                self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5,
                                                          self.cfg.commands.min_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0.,
                                                          self.cfg.commands.max_curriculum)

    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, dtype=torch.float, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = 0.  # commands
        noise_vec[3:6] = noise_scales.ang_vel * \
            noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:9+1*self.num_actions] = noise_scales.dof_pos * \
            noise_level * self.obs_scales.dof_pos                    # p_t
        noise_vec[9+1*self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * \
            noise_level * self.obs_scales.dof_vel  # dp_t
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0.  # a_{t-dt}
        noise_vec[9+3*self.num_actions:11+3 *
                  self.num_actions] = 0.  # clock input
        noise_vec[11+3*self.num_actions:13+3*self.num_actions] = 0.  # theta
        noise_vec[13+3*self.num_actions:14+3 *
                  self.num_actions] = 0.  # gait period
        noise_vec[14+3*self.num_actions:15+3*self.num_actions] = 0.  # b_swing

        # if self.cfg.terrain.measure_heights:
        #     noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements

        return noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(
            self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(
            self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(
            self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, self.num_bodies, -1
        )
        self.feet_state = self.rigid_body_states[:, self.feet_indices, :]
        self.foot_pos = self.rigid_body_states[:, self.feet_indices, :3]
        self.foot_vel = self.rigid_body_states[:, self.feet_indices, 7:10]
        # Periodic Reward Framework
        self.foot_vel_left = self.rigid_body_states[:,
                                                    self.foot_index_left, 7:10]
        self.foot_vel_right = self.rigid_body_states[:,
                                                     self.foot_index_right, 7:10]
        self.foot_pos_left = self.rigid_body_states[:,
                                                    self.foot_index_left, 0:3]
        self.foot_pos_right = self.rigid_body_states[:,
                                                     self.foot_index_right, 0:3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1,
                                                                            3)  # shape: num_envs, num_bodies, xyz axis
        self.foot_contact_force_left = self.contact_forces[:,
                                                           self.foot_index_left, :]
        self.foot_contact_force_right = self.contact_forces[:,
                                                            self.foot_index_right, :]

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.forward_vec = to_torch(
            [1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.p_gains = torch.zeros(
            self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(
            self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.llast_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                         requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                    device=self.device, requires_grad=False)  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
                                           device=self.device, requires_grad=False, )  # TODO change this

        self.base_lin_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(
            self.base_quat, self.gravity_vec)
        self.height_around_feet = torch.zeros( # four points around each foot
            self.num_envs, len(self.feet_indices), 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_relative_heights = torch.zeros(
            self.num_envs, len(self.feet_indices) * 4, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.terrain.measure_heights_actor or self.cfg.terrain.measure_heights_critic:
            self.height_points = self._init_height_points()
            self._calc_point_heights_around_feet()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(
                        f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        # Periodic Reward Framework
        self.theta = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device)
        self.theta[:, 0] = self.cfg.rewards.periodic_reward_framework.theta_left[0]
        self.theta[:, 1] = self.cfg.rewards.periodic_reward_framework.theta_right[0]
        self.gait_time = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device)
        self.phi = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device)
        self.gait_period = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device)
        self.gait_period[:] = self.cfg.rewards.periodic_reward_framework.gait_period[0]
        self.clock_input = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
        )
        self.b_swing = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.b_swing[:] = self.cfg.rewards.periodic_reward_framework.b_swing[0] * 2 * torch.pi
        self.base_height_target = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.base_height_target[:] = self.cfg.rewards.base_height_target
        # obs_history
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_single_obs,
                    dtype=torch.float,
                    device=self.device,
                )
            )
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.single_num_privileged_obs,
                    dtype=torch.float,
                    device=self.device,
                )
            )

    def _calc_point_heights_around_feet(self):
        """ Finds neighboring points around each foot for terrain height measurement."""
        foot_points = self.foot_pos + self.cfg.terrain.border_size
        foot_points = (foot_points/self.cfg.terrain.horizontal_scale).long()
        px_left = foot_points[:, 0, 0].view(-1)
        py_left = foot_points[:, 0, 1].view(-1)
        px_right = foot_points[:, 1, 0].view(-1)
        py_right = foot_points[:, 1, 1].view(-1)
        # clip to the range of height samples
        px_left = torch.clip(px_left, 0, self.height_samples.shape[0]-2)
        py_left = torch.clip(py_left, 0, self.height_samples.shape[1]-2)
        px_right = torch.clip(px_right, 0, self.height_samples.shape[0]-2)
        py_right = torch.clip(py_right, 0, self.height_samples.shape[1]-2)
        # get heights around the feet
        heights1_left = self.height_samples[px_left-1, py_left]  # [x-0.1, y]
        heights2_left = self.height_samples[px_left+1, py_left]  # [x+0.1, y]
        heights3_left = self.height_samples[px_left, py_left-1]  # [x, y-0.1]
        heights4_left = self.height_samples[px_left, py_left+1]  # [x, y+0.1]
        heights1_right = self.height_samples[px_right-1, py_right]  # [x-0.1, y]
        heights2_right = self.height_samples[px_right+1, py_right]  # [x+0.1, y]
        heights3_right = self.height_samples[px_right, py_right-1]  # [x, y-0.1]
        heights4_right = self.height_samples[px_right, py_right+1]  # [x, y+0.1]
        self.height_around_feet[:, 0, 0] = heights1_left * self.terrain.cfg.vertical_scale
        self.height_around_feet[:, 0, 1] = heights2_left * self.terrain.cfg.vertical_scale
        self.height_around_feet[:, 0, 2] = heights3_left * self.terrain.cfg.vertical_scale
        self.height_around_feet[:, 0, 3] = heights4_left * self.terrain.cfg.vertical_scale
        self.height_around_feet[:, 1, 0] = heights1_right * self.terrain.cfg.vertical_scale
        self.height_around_feet[:, 1, 1] = heights2_right * self.terrain.cfg.vertical_scale
        self.height_around_feet[:, 1, 2] = heights3_right * self.terrain.cfg.vertical_scale
        self.height_around_feet[:, 1, 3] = heights4_right * self.terrain.cfg.vertical_scale

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float,
                              device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(
            self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(
            robot_asset)

        self._init_domain_params()

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        print(f"feet names: {feet_names}")
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend(
                [s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend(
                [s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + \
            self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(
            base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1.,
                                        (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(
                rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(
                robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i,
                                                 self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(
                env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(
                env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(
                env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(
            len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                         feet_names[i])
        for i in range(len(feet_names)):
            if "L_Link" in feet_names[i]:
                self.foot_index_left = self.feet_indices[i]
            elif "R_Link" in feet_names[i]:
                self.foot_index_right = self.feet_indices[i]

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.actor_handles[0],
                                                                                      penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.actor_handles[0],
                                                                                        termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(
                0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device),
                                           (self.num_envs / self.cfg.terrain.num_cols), rounding_mode='floor').to(
                torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(
                self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels,
                                                       self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(
                num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(
            self.cfg.domain_rand.push_interval_s / self.dt)
        # Periodic Reward Framework. Constants are init here.
        self.kappa = self.cfg.rewards.periodic_reward_framework.kappa
        self.gait_function_type = self.cfg.rewards.periodic_reward_framework.gait_function_type
        self.a_swing = 0.0
        self.b_stance = 2 * torch.pi

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights_actor and not self.terrain.cfg.measure_heights_critic:
            return

        self.gym.clear_lines(self.viewer)
        sphere_geom = gymutil.WireframeSphereGeometry(
            0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]),
                                           self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym,
                                   self.viewer, self.envs[i], sphere_pose)

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y,
                         device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x,
                         device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points,
                             3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError(
                "Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points),
                                    self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
                self.root_states[:, :3]).unsqueeze(1)

        heights = self._get_terrain_heights_from_points(points)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_terrain_heights_from_points(self, points):
        points = points + self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)
        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        return heights

    def _refresh_rigid_body_states(self):
        # refresh the states of the rigid bodies
        self.foot_vel = self.rigid_body_states[:, self.feet_indices, 7:10]
        self.foot_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        # Periodic Reward Framework
        self.foot_vel_left = self.rigid_body_states[:,
                                                    self.foot_index_left, 7:10]
        self.foot_vel_right = self.rigid_body_states[:,
                                                     self.foot_index_right, 7:10]
        self.foot_pos_left = self.rigid_body_states[:,
                                                    self.foot_index_left, 0:3]
        self.foot_pos_right = self.rigid_body_states[:,
                                                     self.foot_index_right, 0:3]
        self.foot_contact_force_left = self.contact_forces[:,
                                                           self.foot_index_left, :]
        self.foot_contact_force_right = self.contact_forces[:,
                                                            self.foot_index_right, :]

    # ------------ reward functions----------------
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_action_smoothness(self):
        '''Penalize action smoothness'''
        return torch.sum(torch.square(self.actions - 2*self.last_actions + self.llast_actions), dim=-1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_feet_distance(self):
        '''reward for feet distance'''
        feet_xy_distance = torch.norm(
            self.foot_pos_left[:, [0, 1]] - self.foot_pos_right[:, [0, 1]], dim=-1)
        # print(f"feet_xy_distance: {feet_xy_distance}")
        return torch.max(torch.zeros_like(feet_xy_distance),
                         self.cfg.rewards.foot_distance_threshold - feet_xy_distance)

    def _reward_foot_clearance(self):
        """
        Encourage feet to be close to desired height while swinging
        """
        foot_vel_xy_norm = torch.norm(self.foot_vel[:, :, :2], dim=-1)
        clearance_error = torch.sum(
            foot_vel_xy_norm * torch.square(
                self.foot_pos[:, :, 2] - torch.max(self.height_around_feet, dim=-1).values -
                self.cfg.rewards.foot_clearance_target -
                self.cfg.rewards.foot_height_offset
            ), dim=-1
        )
        return torch.exp(-clearance_error / self.cfg.rewards.foot_clearance_tracking_sigma)

    def _reward_survival(self):
        return (~self.reset_buf).float()

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = - \
            (self.dof_pos -
             self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos -
                          self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    # 1. linear velocity tracking
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=-1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    # 2. angular velocity tracking
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    # 3. base height tracking (typically, the target base height is set as a constant)
    def _reward_tracking_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(
            1) - self.measured_heights, dim=1)
        rew = torch.square(base_height - self.cfg.rewards.base_height_target)
        return torch.exp(-rew / self.cfg.rewards.base_height_tracking_sigma)

    # 4. flat orientation
    def _reward_orientation(self):
        # Penalize non flat base orientation
        reward = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return reward

    def _uniped_periodic_gait(self, foot_type):
        # q_frc and q_spd
        if foot_type == "left":
            q_frc = torch.norm(
                self.foot_contact_force_left, dim=-1).view(-1, 1)
            q_spd = torch.norm(
                self.foot_vel_left, dim=-1).view(-1, 1)
            # size: num_envs; need to reshape to (num_envs, 1), or there will be error due to broadcasting
            # modulo phi over 1.0 to get cicular phi in [0, 1.0]
            phi = (self.phi + self.theta[:, 0].unsqueeze(1)) % 1.0
        elif foot_type == "right":
            q_frc = torch.norm(
                self.foot_contact_force_right, dim=-1).view(-1, 1)
            q_spd = torch.norm(
                self.foot_vel_right, dim=-1).view(-1, 1)
            # modulo phi over 1.0 to get cicular phi in [0, 1.0]
            phi = (self.phi + self.theta[:, 1].unsqueeze(1)) % 1.0

        phi *= 2 * torch.pi  # convert phi to radians

        if self.gait_function_type == "smooth":
            # coefficient
            c_swing_spd = 0  # speed is not penalized during swing phase
            c_swing_frc = -1  # force is penalized during swing phase
            c_stance_spd = -1  # speed is penalized during stance phase
            c_stance_frc = 0  # force is not penalized during stance phase

            # clip the value of phi to [0, 1.0]. The vonmises function in scipy may return cdf outside [0, 1.0]
            F_A_swing = torch.clip(torch.tensor(vonmises.cdf(loc=self.a_swing,
                                                             kappa=self.kappa, x=phi.cpu()), device=self.device), 0.0, 1.0)
            F_B_swing = torch.clip(torch.tensor(vonmises.cdf(loc=self.b_swing.cpu(),
                                                             kappa=self.kappa, x=phi.cpu()), device=self.device), 0.0, 1.0)
            F_A_stance = F_B_swing
            F_B_stance = torch.clip(torch.tensor(vonmises.cdf(loc=self.b_stance,
                                                              kappa=self.kappa, x=phi.cpu()), device=self.device), 0.0, 1.0)

            # calc the expected C_spd and C_frc according to the formula in the paper
            exp_swing_ind = F_A_swing * (1 - F_B_swing)
            exp_stance_ind = F_A_stance * (1 - F_B_stance)
            exp_C_spd_ori = c_swing_spd * exp_swing_ind + c_stance_spd * exp_stance_ind
            exp_C_frc_ori = c_swing_frc * exp_swing_ind + c_stance_frc * exp_stance_ind

            # just the code above can't result in the same reward curve as the paper
            # a little trick is implemented to make the reward curve same as the paper
            # first let all envs get the same exp_C_frc and exp_C_spd
            exp_C_frc = -0.5 + (-0.5 - exp_C_spd_ori)
            exp_C_spd = exp_C_spd_ori
            # select the envs that are in swing phase
            is_in_swing = (phi >= self.a_swing) & (phi < self.b_swing)
            indices_in_swing = is_in_swing.nonzero(as_tuple=False).flatten()
            # update the exp_C_frc and exp_C_spd of the envs in swing phase
            exp_C_frc[indices_in_swing] = exp_C_frc_ori[indices_in_swing]
            exp_C_spd[indices_in_swing] = -0.5 + \
                (-0.5 - exp_C_frc_ori[indices_in_swing])

            # Judge if it's the standing gait
            is_standing = (self.b_swing[:] == self.a_swing).nonzero(
                as_tuple=False).flatten()
            exp_C_frc[is_standing] = 0
            exp_C_spd[is_standing] = -1
        elif self.gait_function_type == "step":
            ''' ***** Step Gait Indicator ***** '''
            exp_C_frc = torch.zeros(
                self.num_envs, 1, dtype=torch.float, device=self.device)
            exp_C_spd = torch.zeros(
                self.num_envs, 1, dtype=torch.float, device=self.device)

            swing_indices = (phi >= self.a_swing) & (phi < self.b_swing)
            swing_indices = swing_indices.nonzero(as_tuple=False).flatten()
            stance_indices = (phi >= self.b_swing) & (phi < self.b_stance)
            stance_indices = stance_indices.nonzero(as_tuple=False).flatten()
            exp_C_frc[swing_indices, :] = -1
            exp_C_spd[swing_indices, :] = 0
            exp_C_frc[stance_indices, :] = 0
            exp_C_spd[stance_indices, :] = -1

        return exp_C_spd * q_spd + exp_C_frc * q_frc, \
            exp_C_spd.type(dtype=torch.float), exp_C_frc.type(
                dtype=torch.float)

    def _reward_biped_periodic_gait(self):
        biped_reward_left, self.exp_C_spd_left, self.exp_C_frc_left = self._uniped_periodic_gait(
            "left")
        biped_reward_right, self.exp_C_spd_right, self.exp_C_frc_right = self._uniped_periodic_gait(
            "right")
        # reward for the whole body
        biped_reward = biped_reward_left.flatten() + biped_reward_right.flatten()
        return torch.exp(biped_reward)
