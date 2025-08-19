from legged_gym.envs.base.base_config import BaseConfig

class PointFootEECfg(BaseConfig):
    class env:
        num_envs = 4096
        num_single_obs = 30
        frame_stack = 10   # policy frame stack
        c_frame_stack = 10  # critic frame stack
        num_estimator_features = int(num_single_obs * frame_stack)
        num_estimator_labels = 13
        single_num_privileged_obs = 76
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 6
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        debug_viz = False  # enable debug visualization

    class terrain:
        mesh_type = "plane"  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 0.4
        dynamic_friction = 0.6
        restitution = 0.8
        # rough terrain only:
        measure_heights_actor = False
        measure_heights_critic = False
        measured_points_x = [-0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3]  # 0.6mx0.6m rectangle (without center line)
        measured_points_y = [-0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.1, 0.25, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        curriculum = True
        min_curriculum = -1.0
        max_curriculum = 1.0
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.5, 0.5]  # min max [m/s]
            lin_vel_y = [-0.6, 0.6]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 0.9]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # target angles when action = 0.0
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "foot_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "foot_R_Joint": 0.0,
        }

    class control:
        control_type = 'P'  # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {
            "abad_L_Joint": 40,
            "hip_L_Joint": 40,
            "knee_L_Joint": 40,
            "foot_L_Joint": 0.0,
            "abad_R_Joint": 40,
            "hip_R_Joint": 40,
            "knee_R_Joint": 40,
            "foot_R_Joint": 0.0,
        }  # [N*m/rad]
        damping = {
            "abad_L_Joint": 2.0,
            "hip_L_Joint": 2.0,
            "knee_L_Joint": 2.0,
            "foot_L_Joint": 0.0,
            "abad_R_Joint": 2.0,
            "hip_R_Joint": 2.0,
            "knee_R_Joint": 2.0,
            "foot_R_Joint": 0.0,
        }  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        import os
        import sys
        robot_type = os.getenv("ROBOT_TYPE")

        # Check if the ROBOT_TYPE environment variable is set, otherwise exit with an error
        if not robot_type:
            print("Error: Please set the ROBOT_TYPE using 'export ROBOT_TYPE=<robot_type>'.")
            sys.exit(1)
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/pointfoot/' + robot_type + '/urdf/robot.urdf'
        name = robot_type
        foot_name = 'foot'
        terminate_after_contacts_on = ["base", "abad"]
        penalize_contacts_on = ["knee", "hip"]
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False  # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.0, 1.6]
        randomize_base_mass = True
        added_mass_range = [-1., 2.]
        randomize_base_com = True
        rand_com_vec = [0.03, 0.02, 0.03]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 1.5
        randomize_pd_gain = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]
        randomize_inertia = True
        randomize_inertia_range = [0.9, 1.1]
        randomize_joint_armature = True
        joint_armature_range = [0.11, 0.13]
        randomize_joint_friction = True
        joint_friction_range = [0.00, 0.01]
        randomize_joint_damping = True
        joint_damping_range = [1.4, 1.45]

    class rewards:
        class scales:
            # limitation
            survival = 1
            collision = -1.0
            dof_pos_limits = -10.0
            feet_distance = -100
            # smooth
            action_rate = -0.01
            action_smoothness = -0.01
            ang_vel_xy = -0.05
            lin_vel_z = -0.5
            orientation = -3.0
            dof_acc = -2.5e-07
            dof_vel = -5.e-4
            torques = -2.e-05
            # task
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0
            tracking_base_height = 0.8
            foot_clearance = 0.8
            biped_periodic_gait = 0.5
            foot_landing_vel = -0.15
        
        class periodic_reward_framework:
            '''Periodic reward framework in OSU's paper(https://arxiv.org/abs/2011.01387)'''
            gait_function_type = "step" # can be "step" or "smooth"
            kappa = 20
            resampling_time = 6.0                      # gait resampling time [s]
            gait_period = [0.5]                         # gait period [s]
            num_gaits = 1
            selected_gait = None
            # start of swing is all the same
            b_swing = [0.5]
            theta_left = [0.0]  # front left leg
            theta_right = [0.5]
            swing_phase_ratio = b_swing
            stance_phase_ratio = [1 - swing_phase_ratio[0]]
            
        base_height_target_range = [0.55, 0.8]
        base_height_tracking_sigma = 0.01
        soft_dof_pos_limit = 0.95  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.8
        max_contact_force = 200.  # forces above this value are penalized
        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        foot_distance_threshold = 0.115
        
        foot_clearance_target = 0.06
        foot_height_offset = 0.032
        foot_clearance_tracking_sigma = 0.01
        about_landing_threshold = 0.03
        
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

        clip_observations = 100.
        clip_actions = 100.
        clip_contact_forces = 1000.

    class noise:
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.015
            dof_vel = 0.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1
    
    class joint_pos_des_range:
        hip_joint = [-0.3, 0.3]
        thigh_joint = [-0.5, 0.5]
        calf_joint = [-0.5, 0.5]

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class PointFootEECfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'EERunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        estimator_hidden_dims=[256, 128]
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        
        # estimator params
        estimator_lr = 5.e-4
        num_estimator_epochs = 3

    class runner:
        policy_class_name = 'ActorCriticEE'
        algorithm_class_name = 'PPO_EE'
        num_steps_per_env = 24  # per iteration
        max_iterations = 10000  # number of policy updates

        # logging
        save_interval = 1000  # check for potential saves every this many iterations
        experiment_name = 'pointfoot_ee'
        run_name = 'ee'
        # load and resume
        resume = False
        load_run = "Aug17_17-55-43_ee"  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
