import sys, os
import json
import numpy as np
import random
import signal
from scipy.spatial.transform import Rotation

from omni.isaac.kit import SimulationApp

import argparse
early_parser = argparse.ArgumentParser(add_help=False)
early_parser.add_argument("--headless", action='store_true', help="Run in headless mode")
early_args, _ = early_parser.parse_known_args()

CONFIG = {"headless": early_args.headless}
simulation_app = SimulationApp(CONFIG)

sys.path.append('/home/robot/isaac-sim-4.0/standalone_examples')
from CMES.environment.import_set_file import *
from CMES.environment.object import generate_objects
from CMES.environment.camera import (set_camera, save_camera_infomation, save_image_camera_data, 
                                   save_depth_camera_data, get_camera_bbox_seg_info, 
                                   save_only_camera_infomation, save_contour_data)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

running = True
world = World(stage_units_in_meters=1.0)


def add_box_lighting(world):
    """자연스러운 환경 조명을 추가합니다."""
    try:
        import omni.usd
        from pxr import UsdLux, Gf, UsdGeom
        
        stage = omni.usd.get_context().get_stage()
        main_light_position = np.array([0.0, 0.0, 2.0])
        
        main_light_prim = UsdLux.SphereLight.Define(stage, "/World/environment_light")
        main_light_prim.CreateIntensityAttr().Set(3000.0)
        main_light_prim.CreateRadiusAttr().Set(1.2)
        main_light_prim.CreateColorAttr().Set(Gf.Vec3f(1.0, 0.98, 0.90))
        main_light_prim.CreateExposureAttr().Set(0.1)
        main_light_prim.AddTranslateOp().Set(Gf.Vec3f(main_light_position[0], main_light_position[1], main_light_position[2]))
        
        return True
    except Exception as e:
        print(f"조명 설정 실패: {e}")
        return False


def signal_handler(signum, frame):
    print("프로그램이 종료되었습니다.")
    global running
    running = False


def close_on_enter(event):
    if event.key == 'enter':
        plt.close()


def get_visible_removable_paths(camera):
    """카메라로부터 보이는 제거 가능한 객체들의 경로를 반환합니다."""
    seg_frame = camera.get_current_frame().get("instance_id_segmentation", {})
    seg_info = seg_frame.get("info", {}).get("idToLabels", {})

    visible_paths = set()
    for seg_id, prim_path in seg_info.items():
        if int(seg_id) != 0 and prim_path.startswith("/World/obj/") and "klt" not in prim_path:
             visible_paths.add(prim_path)
             
    return list(visible_paths)


def calculate_6dof_change(initial_pose, current_pose):
    """두 포즈 사이의 6-DOF 변화를 계산합니다."""
    initial_pos, initial_ori = initial_pose
    current_pos, current_ori = current_pose
    
    position_diff = current_pos - initial_pos
    quat_distance = quaternion_distance(current_ori, initial_ori)
    
    try:
        initial_rot = Rotation.from_quat(initial_ori)
        current_rot = Rotation.from_quat(current_ori)
        
        initial_euler = initial_rot.as_euler('xyz', degrees=True)
        current_euler = current_rot.as_euler('xyz', degrees=True)
        
        euler_diff = current_euler - initial_euler
        euler_diff = np.mod(euler_diff + 180, 360) - 180
    except Exception as e:
        euler_diff = np.array([0, 0, 0])
    
    euler_diff_rad = np.radians(euler_diff)
    dof_6_vector = np.concatenate([position_diff, euler_diff_rad])
    
    position_magnitude = np.linalg.norm(position_diff)
    orientation_magnitude = np.linalg.norm(euler_diff_rad)
    
    return {
        "6dof_vector": dof_6_vector.tolist(),
        "position_diff": position_diff.tolist(),
        "orientation_diff_euler": euler_diff.tolist(),
        "position_magnitude": float(position_magnitude),
        "orientation_magnitude": float(orientation_magnitude),
        "quaternion_distance": float(quat_distance),
        "raw_data": {
            "initial_position": initial_pos.tolist(),
            "initial_orientation": initial_ori.tolist(),
            "current_position": current_pos.tolist(),
            "current_orientation": current_ori.tolist()
        }
    }


def quaternion_distance(q1, q2):
    """두 쿼터니언 사이의 거리를 계산합니다."""
    q1_norm = q1 / np.linalg.norm(q1)
    q2_norm = q2 / np.linalg.norm(q2)
    dot_product = np.dot(q1_norm, q2_norm)
    distance = 2 * (1 - abs(dot_product))
    return distance


def get_all_objects_6dof_changes(initial_states, current_primes):
    """모든 객체의 6-DOF 변화를 계산합니다."""
    changes = {}
    
    for prime in current_primes:
        if prime.name in initial_states:
            initial_pose = initial_states[prime.name]
            current_pose = prime.get_world_pose()
            change = calculate_6dof_change(initial_pose, current_pose)
            changes[prime.name] = change
    
    return changes


def physics_based_contact_lifting(target_object, grip_point_local, lift_force=100, max_angle=15):
    """물리 기반 접촉점 들어올리기를 시뮬레이션합니다."""
    try:
        object_pos, object_ori = target_object.get_world_pose()
        contact_world = object_pos + grip_point_local
        force_vector = np.array([0, 0, lift_force])
        
        xy_offset = np.linalg.norm(grip_point_local[:2])
        z_offset = grip_point_local[2]
        
        print(f"    > Applying physics-based lifting:")
        
        if xy_offset < 0.02:
            expected_angle = 0
            stable = True
        else:
            expected_angle = min(xy_offset * 30, max_angle)
            stable = expected_angle < max_angle
        
        return stable, expected_angle
    except Exception as e:
        return False, 0


def lift_object(move_object_path, state, primes, lift_height=0.5, lift_steps=20, steps_per_frame=3, 
                save_intermediate=False, save_paths=None, camera=None, step_offset=0, enable_contact_lifting=True):
    """객체를 점진적으로 들어올립니다."""
    print(f'🤖 Lifting object: {move_object_path}')

    target_object = None
    for p in primes:
        if move_object_path.startswith(p.prim_path):
            target_object = p
            break

    if target_object is None:
        print(f"Warning: Could not find object for path {move_object_path}")
        return

    if target_object.name in state:
        initial_pos, initial_ori = state[target_object.name]
    else:
        initial_pos, initial_ori = target_object.get_world_pose()

    for p in primes:
        if p.name in state:
            pos, ori = state[p.name]
            p.set_world_pose(position=pos, orientation=ori)
    
    contact_point_mode = False
    grip_point_local = np.array([0, 0, 0.03])
    
    if enable_contact_lifting and camera is not None:
        grip_point_world, grip_point_local_detected = find_camera_based_grip_point(camera, move_object_path, target_object)
        if grip_point_world is not None and grip_point_local_detected is not None:
            grip_point_local = grip_point_local_detected
            contact_point_mode = True
        else:
            contact_point_mode = False
    elif not enable_contact_lifting:
        contact_point_mode = False
    else:
        contact_point_mode = False
    
    nearby_objects = []
    for p in primes:
        if p != target_object:
            p_pos, _ = p.get_world_pose()
            distance = np.linalg.norm(p_pos - initial_pos)
            if distance < 0.3:
                nearby_objects.append((p, p_pos))

    for step in range(60):
        world.step(render=True)
    
    original_kinematic_state = None
    try:
        original_kinematic_state = target_object.get_kinematic_enabled()
        target_object.set_kinematic_enabled(True)
    except Exception as e:
        pass

    stable_orientation = initial_ori.copy()
    prev_position = initial_pos.copy()

    for lift_step in range(lift_steps):
        lift_progress = (lift_step + 1) / lift_steps
        eased_progress = lift_progress * lift_progress * (3.0 - 2.0 * lift_progress)
        current_lift = lift_height * eased_progress
        
        new_position = initial_pos + np.array([0, 0, current_lift])
        
        if lift_step > 0:
            alpha = 0.7
            smoothed_position = alpha * new_position + (1 - alpha) * prev_position
        else:
            smoothed_position = new_position
            
        prev_position = smoothed_position
        target_object.set_world_pose(position=smoothed_position, orientation=stable_orientation)
        
        try:
            if hasattr(target_object, 'set_linear_velocity'):
                target_object.set_linear_velocity([0, 0, 0])
            if hasattr(target_object, 'set_angular_velocity'):
                target_object.set_angular_velocity([0, 0, 0])
        except:
            pass
        
        world.step(render=True)
        
        if (lift_step + 1) % 15 == 0 and nearby_objects:
            for p, initial_nearby_pos in nearby_objects:
                current_nearby_pos, _ = p.get_world_pose()
                movement = np.linalg.norm(current_nearby_pos - initial_nearby_pos)
                if movement > 0.01:
                    nearby_objects = [(obj, pos) if obj != p else (obj, current_nearby_pos) for obj, pos in nearby_objects]
        
        if save_intermediate and save_paths and camera and (lift_step + 1) % 5 == 0:
            intermediate_step = step_offset + lift_step
            
            if 'camera_info' in save_paths:
                save_only_camera_infomation(path=save_paths['camera_info'], camera=camera, step=intermediate_step)
            if 'image_data' in save_paths:
                save_image_camera_data(path=save_paths['image_data'], camera=camera, step=intermediate_step)
            if 'depth' in save_paths:
                save_depth_camera_data(path=save_paths['depth'], camera=camera, step=intermediate_step)
            if 'contour' in save_paths:
                save_contour_data(path=save_paths['contour'], camera=camera, step=intermediate_step, next_object_to_remove_path=None)
            if 'normal' in save_paths:
                save_normal_data(path=save_paths['normal'], camera=camera, step=intermediate_step)

    try:
        if original_kinematic_state is not None:
            target_object.set_kinematic_enabled(original_kinematic_state)
    except Exception as e:
        pass
    
    final_target_pos = initial_pos + np.array([3, 0, lift_height])
    
    try:
        target_object.set_kinematic_enabled(True)
    except:
        pass
    
    current_pos, _ = target_object.get_world_pose()
    prev_final_pos = current_pos.copy()
    
    for step in range(30):
        pos_diff = final_target_pos - prev_final_pos
        distance_remaining = np.linalg.norm(pos_diff)
        
        if distance_remaining > 0.02:
            next_pos = prev_final_pos + pos_diff * 0.3
            prev_final_pos = next_pos
            target_object.set_world_pose(position=next_pos, orientation=stable_orientation)
            
            try:
                if hasattr(target_object, 'set_linear_velocity'):
                    target_object.set_linear_velocity([0, 0, 0])
                if hasattr(target_object, 'set_angular_velocity'):
                    target_object.set_angular_velocity([0, 0, 0])
            except:
                pass
        else:
            target_object.set_world_pose(position=final_target_pos, orientation=stable_orientation)
            break
        
        world.step(render=True)
    
    for _ in range(80):
        world.step(render=True)
    
    movement_analysis = get_all_objects_6dof_changes(state, primes)
    return movement_analysis


def save_normal_data(path, camera, step):
    """표면 법선 데이터를 .bin 파일로 저장합니다."""
    frame = camera.get_current_frame()
    normal_data = frame.get("normals")
    if normal_data is not None:
        normal_data = normal_data[:, :, :3]
        file_path = f"{path}/normal_{step}.bin"
        normal_data.astype(np.float32).tofile(file_path)


def find_camera_based_grip_point(camera, target_object_path, target_object):
    """카메라 기반 그립 포인트를 찾습니다."""
    try:
        frame = camera.get_current_frame()
        depth_data = frame.get("distance_to_image_plane")
        seg_frame = frame.get("instance_id_segmentation", {})
        seg_info = seg_frame.get("info", {}).get("idToLabels", {})
        seg_data = seg_frame.get("data")
        
        if depth_data is None or seg_data is None:
            return None, None
        
        object_id = None
        for seg_id, prim_path in seg_info.items():
            if target_object_path.startswith(prim_path) or prim_path.startswith(target_object_path):
                object_id = int(seg_id)
                break
        
        if object_id is None:
            return None, None
        
        object_mask = (seg_data == object_id)
        object_pixels = np.where(object_mask)
        
        if len(object_pixels[0]) == 0:
            return None, None
        
        object_depths = depth_data[object_mask]
        depth_threshold = np.percentile(object_depths, 20)
        top_surface_mask = object_depths <= depth_threshold
        
        if not np.any(top_surface_mask):
            return None, None
        
        top_pixels_y = object_pixels[0][top_surface_mask]
        top_pixels_x = object_pixels[1][top_surface_mask]
        top_depths = object_depths[top_surface_mask]
        
        centroid_u = np.mean(top_pixels_x)
        centroid_v = np.mean(top_pixels_y)
        centroid_depth = np.mean(top_depths)
        
        camera_intrinsics = camera.get_intrinsics_matrix()
        fx = camera_intrinsics[0, 0]
        fy = camera_intrinsics[1, 1] 
        cx = camera_intrinsics[0, 2]
        cy = camera_intrinsics[1, 2]
        
        x_cam = (centroid_u - cx) * centroid_depth / fx
        y_cam = (centroid_v - cy) * centroid_depth / fy
        z_cam = centroid_depth
        
        camera_pos, camera_ori = camera.get_world_pose()
        
        world_x = camera_pos[0] + x_cam
        world_y = camera_pos[1] - y_cam
        world_z = camera_pos[2] - z_cam
        
        grip_point_world = np.array([world_x, world_y, world_z])
        grip_clearance = 0.02
        grip_point_world[2] += grip_clearance
        
        object_pos, object_ori = target_object.get_world_pose()
        grip_point_local = grip_point_world - object_pos
        
        xy_distance = np.linalg.norm(grip_point_local[:2])
        z_distance = abs(grip_point_local[2])
        
        if xy_distance > 0.15:
            return None, None
        
        if z_distance > 0.10:
            return None, None
        
        return grip_point_world, grip_point_local
    except Exception as e:
        return None, None


def analyze_6dof_movements(movement_analysis, threshold_position=0.01, threshold_orientation=0.1):
    """6-DOF 움직임을 분석하고 중요한 변화를 식별합니다."""
    significant_movements = {}
    movement_summary = {
        "total_objects": len(movement_analysis),
        "objects_with_position_change": 0,
        "objects_with_orientation_change": 0,
        "objects_with_significant_change": 0,
        "max_position_change": 0.0,
        "max_orientation_change": 0.0,
        "average_position_change": 0.0,
        "average_orientation_change": 0.0
    }
    
    position_changes = []
    orientation_changes = []
    
    for obj_name, change_data in movement_analysis.items():
        pos_mag = change_data["position_magnitude"]
        ori_mag = change_data["orientation_magnitude"]
        
        position_changes.append(pos_mag)
        orientation_changes.append(ori_mag)
        
        significant_pos = pos_mag > threshold_position
        significant_ori = ori_mag > threshold_orientation
        
        if significant_pos:
            movement_summary["objects_with_position_change"] += 1
        if significant_ori:
            movement_summary["objects_with_orientation_change"] += 1
        if significant_pos or significant_ori:
            movement_summary["objects_with_significant_change"] += 1
            significant_movements[obj_name] = {
                "position_change": pos_mag,
                "orientation_change": ori_mag,
                "6dof_vector": change_data["6dof_vector"],
                "significant_position": significant_pos,
                "significant_orientation": significant_ori
            }
    
    if position_changes:
        movement_summary["max_position_change"] = max(position_changes)
        movement_summary["average_position_change"] = np.mean(position_changes)
    
    if orientation_changes:
        movement_summary["max_orientation_change"] = max(orientation_changes)
        movement_summary["average_orientation_change"] = np.mean(orientation_changes)
    
    return {
        "significant_movements": significant_movements,
        "summary": movement_summary,
        "thresholds": {
            "position_threshold_meters": threshold_position,
            "orientation_threshold_radians": threshold_orientation
        }
    }

def diagnose_object_positions(primes, step_name=""):
    """객체 위치를 진단하여 부유 객체를 식별합니다."""
    klt_bottom = -0.28 + 0.02
    table_surface = 0.72
    
    floating_objects = []
    buried_objects = []
    normal_objects = []
    
    for p in primes:
        try:
            pos, _ = p.get_world_pose()
            height = pos[2]
            
            if height > table_surface + 0.05:
                floating_objects.append((p.name, height))
            elif height < klt_bottom - 0.05:
                buried_objects.append((p.name, height))
            else:
                normal_objects.append((p.name, height))
        except:
            pass
    
    if floating_objects:
        print(f"⚠️ FLOATING objects ({len(floating_objects)}):")
        for name, height in floating_objects:
            print(f"  - {name}: {height:.3f}m")
    
    if buried_objects:
        print(f"⚠️ BURIED objects ({len(buried_objects)}):")
        for name, height in buried_objects:
            print(f"  - {name}: {height:.3f}m")


def make_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--method", type=str, default='mcts')
    parser.add_argument("--min_objects", type=int, default=30)
    parser.add_argument("--max_objects", type=int, default=40)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--render", type=int, default=1)
    parser.add_argument("--save_file", type=str, default='synario_1')
    
    parser.add_argument("--lift_height", type=float, default=0.5, help="Height to lift objects (meters)")
    parser.add_argument("--lift_steps", type=int, default=75, help="Number of steps to complete the lift")
    parser.add_argument("--steps_per_frame", type=int, default=2, help="Physics steps per lift step")
    parser.add_argument("--save_intermediate", action='store_true', help="Save intermediate states during lifting")
    parser.add_argument("--center_point_lifting", action='store_true', help="Use center-based lifting instead of default contact point lifting")
    parser.add_argument("--headless", action='store_true', help="Run in headless mode")
    
    args = parser.parse_args()
    return args


def main(args):
    base_save_path = os.path.join('/home/robot/datasets/isaac_sim_data', args.save_file)
    os.makedirs(base_save_path, exist_ok=True)

    world = World(stage_units_in_meters=1.0)
    
    try:
        from omni.isaac.core.physics_context import PhysicsContext
        physics_context = world.get_physics_context()
        if physics_context:
            physics_context.set_solver_type("TGS")
            physics_context.set_broadphase_type("GPU")
            physics_context.set_gpu_max_rigid_contact_count(1024 * 1024)
            physics_context.set_gpu_max_rigid_patch_count(1024 * 80)
            physics_context.set_gpu_found_lost_pairs_capacity(1024 * 32)
            physics_context.set_gpu_found_lost_aggregate_pairs_capacity(1024 * 32)
            physics_context.set_gpu_total_aggregate_pairs_capacity(1024 * 32)
            physics_context.set_gpu_collision_stack_size(1024 * 64)
            physics_context.set_gpu_heap_capacity(1024 * 64)
            physics_context.set_gpu_temp_buffer_capacity(1024 * 16)
            physics_context.set_gpu_max_num_partitions(8)
            physics_context.set_physics_dt(1.0/60.0)
    except Exception as e:
        pass
    
    signal.signal(signal.SIGINT, signal_handler)
    
    for episode in range(args.num_episodes):
        if not running:
            break
        
        print(f"--- Starting Episode {episode+1}/{args.num_episodes} ---")

        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath("/World/obj"):
            stage.RemovePrim("/World/obj")

        camera = set_camera()
        world.scene.add(camera)
        # add_box_lighting(world)  # 조명 비활성화 - Simple_Room 기본 조명만 사용

        episode_save_path = os.path.join(base_save_path, f"episode_{episode}")
        CAMERA_SAVE_PATH = os.path.join(episode_save_path, 'camera_info')
        CAMERA_IMAGE_SAVE_PATH = os.path.join(episode_save_path, 'image_data')
        CAMERA_DEPTH_SAVE_PATH = os.path.join(episode_save_path, 'depth')
        CONTOUR_SAVE_PATH = os.path.join(episode_save_path, 'contour')
        NORMAL_SAVE_PATH = os.path.join(episode_save_path, 'normal')
        ACTION_METADATA_SAVE_PATH = os.path.join(episode_save_path, 'action_metadata')

        os.makedirs(CAMERA_SAVE_PATH, exist_ok=True)
        os.makedirs(CAMERA_IMAGE_SAVE_PATH, exist_ok=True)
        os.makedirs(CAMERA_DEPTH_SAVE_PATH, exist_ok=True)
        os.makedirs(CONTOUR_SAVE_PATH, exist_ok=True)
        os.makedirs(NORMAL_SAVE_PATH, exist_ok=True)
        os.makedirs(ACTION_METADATA_SAVE_PATH, exist_ok=True)
        
        intermediate_save_paths = None
        if args.save_intermediate:
            intermediate_save_paths = {
                'camera_info': CAMERA_SAVE_PATH,
                'image_data': CAMERA_IMAGE_SAVE_PATH,
                'depth': CAMERA_DEPTH_SAVE_PATH,
                'contour': CONTOUR_SAVE_PATH,
                'normal': NORMAL_SAVE_PATH
            }
        
        primes = generate_objects(world, min_objects=args.min_objects, max_objects=args.max_objects)
        world.reset()
        
        print("🔄 Settling objects...")
        
        for step in range(150):
            world.step(render=True)
        
        try:
            for p in primes:
                try:
                    p.set_linear_damping(0.95)
                    p.set_angular_damping(0.95)
                except:
                    pass
            
            for step in range(100):
                world.step(render=True)
            
            for p in primes:
                try:
                    p.set_linear_damping(0.8)
                    p.set_angular_damping(0.8)
                except:
                    pass
        except Exception as e:
            pass
        
        for step in range(150):
            world.step(render=True)
        
        try:
            for p in primes:
                try:
                    current_pos, current_ori = p.get_world_pose()
                    expected_bottom = -0.28 + 0.02
                    if current_pos[2] > expected_bottom + 0.05:
                        settle_pos = current_pos.copy()
                        settle_pos[2] = max(settle_pos[2] - 0.01, expected_bottom)
                        p.set_world_pose(position=settle_pos, orientation=current_ori)
                except:
                    pass
            
            for step in range(100):
                world.step(render=True)
        except Exception as e:
            pass
        
        try:
            klt_bbox_check = compute_combined_aabb(prim_paths=["/World/obj/klt"])
            if klt_bbox_check:
                bin_min_x, bin_min_y, bin_min_z, bin_max_x, bin_max_y, bin_max_z = klt_bbox_check
                
                safety_margin = 0.03
                safe_min_x = bin_min_x + safety_margin
                safe_max_x = bin_max_x - safety_margin
                safe_min_y = bin_min_y + safety_margin
                safe_max_y = bin_max_y - safety_margin
                safe_min_z = bin_min_z + 0.01
                
                corrected_objects = 0
                for p in primes:
                    try:
                        current_pos, current_ori = p.get_world_pose()
                        corrected_pos = current_pos.copy()
                        needs_correction = False
                        
                        if current_pos[0] < safe_min_x:
                            corrected_pos[0] = safe_min_x
                            needs_correction = True
                        elif current_pos[0] > safe_max_x:
                            corrected_pos[0] = safe_max_x
                            needs_correction = True
                            
                        if current_pos[1] < safe_min_y:
                            corrected_pos[1] = safe_min_y
                            needs_correction = True
                        elif current_pos[1] > safe_max_y:
                            corrected_pos[1] = safe_max_y
                            needs_correction = True
                            
                        if current_pos[2] < safe_min_z:
                            corrected_pos[2] = safe_min_z
                            needs_correction = True
                        
                        if needs_correction:
                            p.set_world_pose(position=corrected_pos, orientation=current_ori)
                            
                            try:
                                if hasattr(p, 'set_linear_velocity'):
                                    p.set_linear_velocity([0, 0, 0])
                                if hasattr(p, 'set_angular_velocity'):
                                    p.set_angular_velocity([0, 0, 0])
                            except:
                                pass
                                
                            corrected_objects += 1
                    except:
                        pass
                
                if corrected_objects > 0:
                    print(f"📍 Corrected {corrected_objects} objects outside bin boundaries")
                
                for step in range(50):
                    world.step(render=True)
        except Exception as e:
            pass
        
        diagnose_object_positions(primes, "After Settling")

        try:
            klt_bbox = compute_combined_aabb(prim_paths=["/World/obj/klt"])
            klt_min_x, klt_min_y, _, klt_max_x, klt_max_y, _ = klt_bbox

            objects_inside_bin = []
            objects_outside_bin = []

            for p in primes:
                pos, _ = p.get_world_pose()
                if klt_min_x <= pos[0] <= klt_max_x and klt_min_y <= pos[1] <= klt_max_y:
                    objects_inside_bin.append(p)
                else:
                    objects_outside_bin.append(p)

            if objects_outside_bin:
                print(f"🗑️ Removing {len(objects_outside_bin)} objects outside bin")
                for p_out in objects_outside_bin:
                    p_out.set_world_pose(position=p_out.get_world_pose()[0] + np.array([5, 0, 0]))
                
                for _ in range(60):
                    world.step(render=True)
            
            target_primes = objects_inside_bin
            diagnose_object_positions(target_primes, "After Cleanup")

        except Exception as e:
            target_primes = primes

        count_step = 0
        
        while running:
            visible_object_paths = get_visible_removable_paths(camera)
            target_prim_paths_set = {p.prim_path for p in target_primes}
            valid_visible_paths = [path for path in visible_object_paths if any(path.startswith(tp) for tp in target_prim_paths_set)]
            next_action_path = random.choice(valid_visible_paths) if valid_visible_paths else None

            print(f"📸 Step {count_step}: {next_action_path}")
            save_only_camera_infomation(path=CAMERA_SAVE_PATH, camera=camera, step=count_step)
            save_image_camera_data(path=CAMERA_IMAGE_SAVE_PATH, camera=camera, step=count_step)
            save_depth_camera_data(path=CAMERA_DEPTH_SAVE_PATH, camera=camera, step=count_step)
            save_contour_data(path=CONTOUR_SAVE_PATH, camera=camera, step=count_step, next_object_to_remove_path=next_action_path)
            save_normal_data(path=NORMAL_SAVE_PATH, camera=camera, step=count_step)

            if next_action_path is None:
                print(f"✅ Episode {episode} complete - all objects removed")
                break

            state = {p.name: p.get_world_pose() for p in target_primes}
            step_offset = count_step * 1000
            
            movement_analysis = lift_object(
                move_object_path=next_action_path, 
                state=state, 
                primes=target_primes,
                lift_height=args.lift_height,
                lift_steps=args.lift_steps,
                steps_per_frame=args.steps_per_frame,
                save_intermediate=args.save_intermediate,
                save_paths=intermediate_save_paths,
                camera=camera,
                step_offset=step_offset,
                enable_contact_lifting=not args.center_point_lifting
            )
            
            # lifted_objects에서 물체 이름만 추출 (visualize_contours.py와 동일한 로직)
            lifted_object_names = []
            lifted_objects = [next_action_path]  # action_metadata의 lifted_objects와 동일하게 구성
            for lifted_path in lifted_objects:
                if '/obj/' in lifted_path:
                    obj_name = lifted_path.split('/obj/')[1].split('/')[0]
                    lifted_object_names.append(obj_name)
            
            # 움직임 기준에 따라 moved_objects 생성 (lifted 객체 제외)
            moved_objects = []
            for obj_name, change_data in movement_analysis.items():
                # lifted 객체는 moved_objects에서 제외 (visualize_contours.py와 동일한 로직)
                if obj_name in lifted_object_names:
                    continue
                    
                pos_change = change_data["position_magnitude"]
                ori_change = change_data["orientation_magnitude"]
                
                # 임계값 체크: pos > 0.0175m 또는 ori > 0.13rad
                if pos_change > 0.0175 or ori_change > 0.13:
                    # visualize_contours.py와 동일한 형식으로 상세 정보 저장
                    moved_object_info = {
                        "object_name": obj_name,
                        "position_magnitude": pos_change,
                        "orientation_magnitude": ori_change,
                        "position_diff": change_data.get('position_diff', [0.0, 0.0, 0.0]),
                        "orientation_diff_euler": change_data.get('orientation_diff_euler', [0.0, 0.0, 0.0]),
                        "6dof_vector": change_data.get('6dof_vector', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    }
                    moved_objects.append(moved_object_info)
            
            # moved_objects를 움직임 크기 순으로 정렬 (visualize_contours.py와 동일)
            moved_objects.sort(
                key=lambda x: x['position_magnitude'] + x['orientation_magnitude'], 
                reverse=True
            )
            
            action_metadata = {
                "step": count_step,
                "timestamp": None,
                "lifted_objects": lifted_objects,  # 위에서 정의한 lifted_objects 사용
                "moved_objects": moved_objects,
                "num_objects_remaining": len(target_primes) - 1,
                "movement_analysis": movement_analysis,
                "additional_info": {
                    "lift_height": args.lift_height,
                    "lift_steps": args.lift_steps,
                    "contact_point_lifting": not args.center_point_lifting,
                    "step_offset": step_offset,
                    "movement_thresholds": {
                        "position_threshold": 0.0175,
                        "orientation_threshold": 0.13
                    }
                }
            }
            
            action_metadata_path = os.path.join(ACTION_METADATA_SAVE_PATH, f'action_metadata_{count_step}.json')
            with open(action_metadata_path, 'w') as f:
                json.dump(action_metadata, f, indent=2)
            
            analysis_result = analyze_6dof_movements(movement_analysis)
            summary = analysis_result["summary"]
            significant = analysis_result["significant_movements"]
            
            if significant:
                print(f"📊 {len(significant)} objects moved significantly")
                for obj_name, data in significant.items():
                    print(f"  • {obj_name}: {data['position_change']:.3f}m, {data['orientation_change']:.3f}rad")
            
            count_step += 1

            for _ in range(80):
                world.step(render=True)

        if running:
            world.clear()
            print(f"🔄 Episode {episode+1} completed")

    simulation_app.close()
    print("Simulation app closed")


if __name__ == "__main__":
    args = make_arg()
    main(args)