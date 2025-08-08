from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core import World
from omni.isaac.core.prims import XFormPrim, GeometryPrim, RigidPrim,ClothPrim
from omni.isaac.core.objects import FixedCuboid, DynamicCuboid
import numpy as np
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import omni.usd
from omni.isaac.core.utils.semantics import add_update_semantics
import omni.isaac.core.utils.prims as prims_utils

import json
import os
import random

script_dir = os.path.dirname(os.path.abspath(__file__))

def generate_objects(world, min_objects=30, max_objects=40):
    return_objects_primes=[]
    obj_prim = world.scene.add(
    XFormPrim(prim_path="/World/obj",
            name="obj_prim",
            translation=np.array([0.0, 0.61, 1]),
            orientation=euler_angles_to_quat(np.array([0, 0, 0]), degrees=True),
            )
    )
    #return_objects_primes.append(obj_prim)

    ur5_2f140_usd_path = os.path.join(script_dir, '0201_robot_base.usd')
    add_reference_to_stage(ur5_2f140_usd_path, '/World')

    add_reference_to_stage(usd_path=os.path.join(script_dir, 'table.usd'),
                        prim_path='/World/table')
    table_prim = world.scene.add(XFormPrim('/World/table', name='table',
                                        translation=np.array([0.0, 0.61, 0]),orientation=np.array(euler_angles_to_quat(np.array([0.0,0.0,np.pi/2])))))
    #return_objects_primes.append(table_prim)
    table_upper_prim = world.scene.add(FixedCuboid('/World/table/upper', name='table2',
                                    translation=np.array([0, 0, 0.72]),scale=np.array([0.7,1.2,0.001])))
    #return_objects_primes.append(table_upper_prim)
    env=add_reference_to_stage(usd_path=os.path.expanduser('~/Downloads/isaac_sim-assets-1-2023.1.0/Assets/Isaac/2023.1.0/Isaac/Environments/Simple_Room/simple_room.usd'),prim_path='/World/Env')
    simple_room_prim = world.scene.add(XFormPrim('/World/Env', name='env',translation=np.array([0,0,0.77])))
    #return_objects_primes.append(simple_room_prim)
    
    # KLT 박스 생성 - Enhanced height to prevent objects from bouncing out
    add_reference_to_stage(usd_path=os.path.expanduser("~/Downloads/isaac_sim-assets-2-2023.1.0/Assets/Isaac/2023.1.0/Isaac/Props/KLT_Bin/small_KLT.usd"), prim_path="/World/obj/klt")
    klt_bin_box = world.scene.add(
                RigidPrim(prim_path="/World/obj/klt", name="klt",translation=np.array([0.00, 0.0, 0.72]),scale=np.array([2.3, 2.5, 2.0]),orientation=np.array(euler_angles_to_quat(np.array([0.0,0.0,np.pi/2])))),
            )
    
    # KL bin을 안정적으로 고정하되 초기 배치는 자연스럽게 설정
    try:
        # 원래 설정에서 질량만 적당히 올려서 안정화
        klt_bin_box.set_mass(50.0)  # 적당히 무겁게 설정 (기존 10kg → 50kg)
        klt_bin_box.set_friction_coefficient(0.9)  # 높은 마찰로 미끄러짐 방지 (원래 설정 유지)
        
        print(f"  > KL bin set with stable physics (50kg mass)")
        
    except Exception as e:
        print(f"  > Warning: Could not set KL bin physics: {e}")
    
    # Calculate proper KLT bin bottom height for realistic object placement
    klt_center_height = 0.72  # KLT bin center at table surface (0.72m) - 테이블 위에 올바르게 배치
    klt_z_scale = 2.0         # Updated Z-axis scale factor for reduced height
    # Assuming KLT bin original height is ~20cm, scaled to 40cm (reduced from 68cm)
    # Bottom would be at center - (scaled_height/2)
    estimated_klt_height = 0.40  # 40cm scaled height - reduced for better proportions
    klt_bottom_height = klt_center_height - (estimated_klt_height / 2)
    # Add small offset above bottom for natural resting
    object_spawn_height = klt_bottom_height + 0.02  # 2cm above bin bottom
    
    print(f"Stable KLT Bin Setup:")
    print(f"  > Center height: {klt_center_height:.3f}m")
    print(f"  > Reduced height: {estimated_klt_height:.3f}m")
    print(f"  > Estimated bottom: {klt_bottom_height:.3f}m")
    print(f"  > Object spawn height: {object_spawn_height:.3f}m")
    print(f"  > Bin walls now {estimated_klt_height*100:.0f}cm tall - stable placement")
    
    #return_objects_primes.append(klt_bin_box)
    
    # Enhanced object placement system for more realistic distribution
    # Calculate proper bin boundaries based on actual KLT bin size
    bin_x_range = 0.22  # Increased from 0.18 for better coverage (44cm total)
    bin_y_range = 0.18  # Increased from 0.1 for better coverage (36cm total)
    
    # Create initial point with slight bias toward center
    x = round((0 + np.random.randn()/12), 3)  # Reduced variance for more central tendency
    y = round((0 + np.random.randn()/15), 3)  # Reduced variance for more central tendency
    
    # Enhanced height system - create natural dropping and stacking
    base_drop_height = object_spawn_height + 0.08  # Start 8cm above bin bottom for natural drop
    height_variance = 0.12  # Increased to 12cm for more natural layering and overlap
    
    points = [np.array([x, y, base_drop_height + np.random.uniform(0, height_variance)])]
    
    # Generate more spawn points than needed, then select best distribution
    num_points_to_generate = 50  # Increased from 45 for better selection
    
    print(f"Enhanced Object Placement Setup:")
    print(f"  > Bin coverage: X=±{bin_x_range:.2f}m ({bin_x_range*2:.2f}m), Y=±{bin_y_range:.2f}m ({bin_y_range*2:.2f}m)")
    print(f"  > Drop height: {base_drop_height:.3f}m + {height_variance:.2f}m variance")
    print(f"  > Expected natural stacking and overlap")
    
    for _ in range(num_points_to_generate - 1):
        # Adaptive minimum distance based on realistic object spacing
        # Use smaller minimum distance to allow for more realistic dense packing
        adaptive_min_distance = 0.06  # Reduced from 0.10 (6cm allows closer packing)
        
        # Enhanced drop height with more variance for natural falling patterns
        drop_z = base_drop_height + np.random.uniform(0, height_variance)
        
        new_point = generate_random_point_with_min_distance(
            points, 
            -bin_x_range, bin_x_range,    # Enhanced X range 
            -bin_y_range, bin_y_range,    # Enhanced Y range
            drop_z - 0.02, drop_z + 0.02, # Small Z variance around drop height
            adaptive_min_distance         # Adaptive minimum distance
        )
        points.append(new_point)
    
    # Shuffle for random assignment but maintain good distribution
    random.shuffle(points)
    
    print(f"  > Generated {len(points)} spawn points with adaptive spacing")
    
    ############### generate object
    with open(os.path.join(script_dir, 'object_information.json'),'r') as f:
        object_data = json.load(f)

    # --- New generation logic based on min/max ---
    objects_to_create_counts = {}
    available_object_names = list(object_data.keys())
    # Extract weights, defaulting to 1 if not specified
    object_weights = [object_data[name].get("weight", 1) for name in available_object_names]

    # 1. Guarantee the minimum number of objects
    # We use `random.choices` which allows for duplicates and respects weights
    guaranteed_objects = random.choices(available_object_names, weights=object_weights, k=min_objects)
    for name in guaranteed_objects:
        objects_to_create_counts[name] = objects_to_create_counts.get(name, 0) + 1

    # 2. Randomly add more objects up to the maximum
    num_to_potentially_add = max_objects - min_objects
    for _ in range(num_to_potentially_add):
        # Add an additional object with a 50% probability
        if random.random() < 0.5:
            # `random.choices` returns a list, so we take the first element
            additional_object_name = random.choices(available_object_names, weights=object_weights, k=1)[0]
            objects_to_create_counts[additional_object_name] = objects_to_create_counts.get(additional_object_name, 0) + 1
    
    # Ensure we don't try to create more objects than we have spawn points for
    total_objects_to_create = sum(objects_to_create_counts.values())
    if total_objects_to_create > len(points):
        print(f"Warning: Trying to create {total_objects_to_create} objects, but only {len(points)} spawn points available. Some objects will not be created.")
        # This part is tricky. We need to reduce the count of objects to fit.
        # A simple approach is to randomly decrement counts until it fits.
        while sum(objects_to_create_counts.values()) > len(points):
            # Pick a random object that has at least one instance to be created
            random_object_to_reduce = random.choice([k for k, v in objects_to_create_counts.items() if v > 0])
            objects_to_create_counts[random_object_to_reduce] -= 1

    points_idx = 0
    for name, count in objects_to_create_counts.items():
        if name not in object_data or count == 0:
            continue
        data_info = object_data[name]
        for i in range(count):
            # Each instance gets a unique prim path
            prim_path = f'/World/obj/{name}_{i}'
            
            # The semantic label is set to the UNIQUE prim path
            prims_utils.create_prim(
                prim_path=prim_path,
                prim_type="Xform",
                usd_path=data_info["object_path"],
                semantic_label=prim_path, # CRITICAL CHANGE
                semantic_type="class"
            )
            
            # The name for the RigidPrim must also be unique for the state dictionary
            unique_name = f'{name}_{i}'
            prim = world.scene.add(RigidPrim(
                prim_path=prim_path, 
                name=unique_name, 
                mass=data_info["mass"]
            ))
            
            # Add enhanced physics properties for ultra-realistic, non-bouncy interactions
            try:
                # Apply mass-based physics tuning for more realistic behavior
                object_mass = data_info["mass"]
                
                # Base physics properties — tuned for realistic behavior
                if object_mass < 0.2:  # Very light objects (markers, small items)
                    contact_stiffness = 800     # Reduced from 1500 to prevent penetration
                    contact_damping   = 400     # Reduced accordingly
                    restitution       = 0.01    # 거의 튕기지 않음
                    friction          = 0.7     # Increased friction for stability
                    rolling_friction  = 0.03    # Increased for stability

                elif object_mass < 0.5:  # Light objects (cans, small boxes)
                    contact_stiffness = 1000    # Reduced from 2000
                    contact_damping   = 500     # Reduced accordingly
                    restitution       = 0.02
                    friction          = 0.7     # Increased
                    rolling_friction  = 0.04

                elif object_mass < 1.0:  # Medium objects (boxes, bottles)
                    contact_stiffness = 1200    # Reduced from 2500
                    contact_damping   = 600     # Reduced accordingly
                    restitution       = 0.03
                    friction          = 0.65    # Increased
                    rolling_friction  = 0.05

                else:  # Heavy objects (large containers)
                    contact_stiffness = 1500    # Reduced from 3000
                    contact_damping   = 700     # Reduced accordingly
                    restitution       = 0.04
                    friction          = 0.6     # Increased
                    rolling_friction  = 0.06
                
                # Apply the calculated physics properties
                prim.set_friction_coefficient(friction)
                prim.set_rolling_friction_coefficient(rolling_friction)
                prim.set_restitution(restitution)  # Ultra-low bounce for realism
                
                # Apply ultra-realistic contact properties
                try:
                    prim.set_contact_stiffness(contact_stiffness)  # Much lower for soft contacts
                    prim.set_contact_damping(contact_damping)      # Higher damping for energy absorption
                    
                    # Enhanced sleep settings for faster stabilization
                    prim.set_sleep_threshold(0.005)     # Lower threshold for faster sleep
                    prim.set_stabilization_threshold(0.0005)  # Even lower for better stability
                    
                    print(f"  > Applied ultra-realistic physics to {unique_name}: mass={object_mass}kg, contact_stiffness={contact_stiffness}, friction={friction}, restitution={restitution}")
                except:
                    print(f"  > Applied basic realistic physics to {unique_name}: mass={object_mass}kg, friction={friction}, restitution={restitution}")
                
                # Special handling for cylindrical objects (cans, bottles) - prevent excessive rolling
                if any(keyword in name.lower() for keyword in ['can', 'bottle', 'cylinder']):
                    try:
                        prim.set_rolling_friction_coefficient(min(0.8, rolling_friction + 0.3))  # Extra rolling resistance
                        print(f"    > Extra rolling resistance applied to cylindrical object {unique_name}")
                    except:
                        pass
                        
            except Exception as e:
                print(f"  > Warning: Could not set physics properties for {unique_name}: {e}")
            
            world.scene.add(GeometryPrim(
                prim_path=prim_path, 
                name=f'{unique_name}_geom', 
                translation=np.array([points[points_idx][0], points[points_idx][1], points[points_idx][2]]),
                orientation=euler_angles_to_quat(np.array(data_info["orientation"])),
                scale=np.array(data_info["scale"]),
                collision=True
            ))
            return_objects_primes.append(prim)
            points_idx += 1

    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Cube":
            label = str(prim.GetPrimPath()).split("/")[-1]
            add_update_semantics(prim, semantic_label=label, type_label="class")
    

    return return_objects_primes




def generate_random_point_with_min_distance(existing_points, x_min, x_max, y_min, y_max, z_min, z_max, min_distance):
    max_attempts = 1000  # 최대 시도 횟수
    for _ in range(max_attempts):
        # 랜덤한 새로운 점 생성 (x, y, z 모두 고려)
        new_point = np.array([
            round(np.random.uniform(x_min, x_max),3),
            round(np.random.uniform(y_min, y_max),3),
            round(np.random.uniform(z_min, z_max),3)
        ])
        
        # 모든 기존 점들과 최소 거리 이상 떨어져 있는지 확인 (3차원 거리 계산)
        if all(np.linalg.norm(new_point - point) >= min_distance for point in existing_points):
            return new_point
    
    raise ValueError("새로운 점을 생성할 수 없습니다. (시도 횟수 초과)")
