from omni.isaac.sensor import Camera
import numpy as np
from omni.isaac.core.utils.rotations import euler_angles_to_quat
#from sensor_msgs.msg import Image
from PIL import Image
import sys,os
import json
import cv2

def set_camera():
    # Resolution options:
    # - (1224, 1024) - Current: 1.195:1 aspect ratio (slightly wider than square)
    # - (1280, 720) - Original: 16:9 aspect ratio (standard widescreen)
    # - (1920, 1080) - High definition: 16:9 aspect ratio
    # - (1600, 900) - Alternative: 16:9 aspect ratio
    camera1 = Camera(
        prim_path= '/World/camera/realsense_camera1',
        resolution=(1224, 1024),  # Change this to (1280, 720) for original aspect ratio
    )

    camera1.set_world_pose(
        position=np.array([0.0, 0.61, 1.58]),  # Z lowered from 1.8 to 1.6
        orientation=np.array(euler_angles_to_quat(np.array([0,0,0]))),
        camera_axes="usd"
    )
    camera1.initialize()
    camera1.set_focal_length(0.23) # Keep focal length
    camera1.set_focus_distance(4.0) # Keep focus distance
    
    # Adjust apertures for 1224x1024 resolution to maintain proper FOV
    # Calculate apertures to maintain similar field of view as 1280x720
    # Original 1280x720 with apertures (0.1973, 0.148) had aspect ratio 1.78:1
    # New 1224x1024 has aspect ratio 1.195:1
    
    # Increase aperture values to widen field of view (make objects appear smaller)
    # Larger aperture = wider field of view = objects appear smaller
    camera1.set_horizontal_aperture(0.28)  # Increased from 0.1973 to widen horizontal FOV
    
    # Adjust vertical aperture based on new aspect ratio and wider FOV
    # vertical_aperture = horizontal_aperture * (height/width)
    new_vertical_aperture = 0.28 * (1024 / 1224)
    camera1.set_vertical_aperture(new_vertical_aperture)  # Calculated: ~0.234
    
    # print(f"Camera settings updated for 1224x1024 with wider FOV:")
    # print(f"  - Horizontal aperture: 0.28 (increased from 0.1973)")
    # print(f"  - Vertical aperture: {new_vertical_aperture:.4f} (calculated)")
    # print(f"  - Aspect ratio: {1224/1024:.3f}:1")
    # print(f"  - Effect: Wider field of view, objects will appear smaller")
    # print(f"  - Camera position: [0.0, 0.61, 1.8] (Z raised from 1.5 to 1.8)")
    # print(f"  - Camera raised by: {1.8-1.5:.1f}m (Y axis kept at original 0.61)")
    
    camera1.set_clipping_range(0.01, 10000.0)
    #camera1.add_pointcloud_to_frame()
    # camera1.add_motion_vectors_to_frame()
    camera1.add_distance_to_image_plane_to_frame()
    camera1.add_bounding_box_2d_loose_to_frame()
    camera1.add_instance_id_segmentation_to_frame()
    camera1.add_normals_to_frame()
    #camera1.add_bounding_box_3d_to_frame()
    return camera1

def get_camera_contour_data(camera):
    """
    Processes the instance segmentation mask to extract object contours and
    matches them with their corresponding ID and prim path.
    Only keeps the largest contour for each object to avoid fragmentation.
    """
    seg_frame = camera.get_current_frame().get("instance_id_segmentation", {})
    seg_data = seg_frame.get("data")
    seg_info = seg_frame.get("info", {}).get("idToLabels", {})

    if seg_data is None:
        return {"contours": []}

    unique_ids = np.unique(seg_data)
    contours_list = []

    for obj_id in unique_ids:
        obj_id = int(obj_id)
        
        # The semantic label is now the prim path
        prim_path = seg_info.get(str(obj_id), "unknown")

        # Filter out anything that is not a controllable object under /World/obj/
        # and also filter out the bin itself.
        if not prim_path.startswith("/World/obj/") or "klt" in prim_path:
            continue

        # Create a binary mask for the current object
        binary_mask = (seg_data == obj_id).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Only keep the largest contour for this object to avoid fragmentation
        if contours:
            # Find the contour with the largest area
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            
            # Skip very small contours (likely noise)
            if contour_area < 10:  # Skip contours smaller than 10 pixels
                continue
            
            # Approximate the contour to have fewer points
            epsilon = 0.005 * cv2.arcLength(largest_contour, True)
            approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

            if approx_contour.size >= 2:  # Valid contour
                contour_points = approx_contour.squeeze(axis=1).tolist()
                
                contour_info = {
                    "object_id": obj_id,
                    "prim_path": prim_path,
                    "contour_points": contour_points,
                    "area": float(contour_area),
                    "num_fragments_found": len(contours),  # Debug info: how many fragments were found
                    "largest_fragment_selected": True  # Indicates we selected the largest fragment
                }
                contours_list.append(contour_info)
                
                # Debug log for objects with multiple fragments
                if len(contours) > 1:
                    print(f"  🔍 Object {prim_path}: Found {len(contours)} fragments, selected largest (area: {contour_area:.1f})")

    return {"contours": contours_list}

def save_contour_data(path, camera, step, next_object_to_remove_path=None):
    """
    Saves the extracted contour data to a JSON file, including the
    prim path of the next object to be removed.
    """
    contour_data = get_camera_contour_data(camera)
    contour_data["next_action_prim_path"] = next_object_to_remove_path
    contour_path = os.path.join(path, f'contour_{step}.json')
    
    with open(contour_path, 'w') as f:
        json.dump(contour_data, f, indent=4)

def save_camera_infomation(path,camera):
    print('---save_camera_infomation---')
    rgb_data=camera.get_rgba()[:, :, :3]
    image1 = Image.fromarray(rgb_data, 'RGB')
    save_path=os.path.join(path,f'camera.png')
    image1.save(save_path)

    ######depth_data저장
    depth_data=camera.get_depth()
    # print('depth_data',depth_data)
    depth_save_path=os.path.join(path,f'camera_depth.npy')
    np.save(depth_save_path,depth_data) 

    ############bbox 저장 ###########
    bbox_path=os.path.join(path,f'camera_bbox.json')
    bbox_data=camera.get_current_frame()['bounding_box_2d_loose']
    bbox_data['data'] = bbox_data['data']
    if isinstance(bbox_data['data'], np.ndarray):
        bbox_data['data'] = bbox_data['data'].tolist()
    if isinstance(bbox_data['info']['bboxIds'], np.ndarray):
        bbox_data['info']['bboxIds'] = bbox_data['info']['bboxIds'].tolist()
    with open(bbox_path, 'w') as json_file:
        json.dump(bbox_data, json_file)


    ################################### segmentation 저장
    seg_data=camera.get_current_frame()['instance_id_segmentation']['data']
    seg_info=camera.get_current_frame()['instance_id_segmentation']['info']
    bbox_data['Segmentation_info'] = seg_info
    seg_save_path = os.path.join(path,f'camera_segmentation.npy')
    np.save(seg_save_path,seg_data)

    ###############camera feature 저장
    camera_intrinsic=camera.get_intrinsics_matrix().tolist()
    camera_focal=camera.get_focal_length()
    camera_focus=camera.get_focus_distance()
    camera_horizontal=camera.get_horizontal_aperture()
    camera_aperture=camera.get_lens_aperture()
    camera_tuple={
        'camera_intrinsic' : camera_intrinsic,
        'camera_focal' : camera_focal,
        'camera_focus' : camera_focus,
        'camera_horizontal' : camera_horizontal,
        'camera_aperture' : camera_aperture
    }   
    camera_tuple_path=os.path.join(path,f'camera_information.json')
    with open(camera_tuple_path, 'w') as json_file:
        json.dump(camera_tuple, json_file)


def save_only_camera_infomation(path,camera,step):
    ############bbox 저장 ###########
    bbox_path=os.path.join(path,f'camera_bbox_{step}.json')
    bbox_data=camera.get_current_frame()['bounding_box_2d_loose']
    bbox_data['data'] = bbox_data['data']
    if isinstance(bbox_data['data'], np.ndarray):
        bbox_data['data'] = bbox_data['data'].tolist()
    if isinstance(bbox_data['info']['bboxIds'], np.ndarray):
        bbox_data['info']['bboxIds'] = bbox_data['info']['bboxIds'].tolist()
    with open(bbox_path, 'w') as json_file:
        json.dump(bbox_data, json_file)


def save_image_camera_data(path,camera,step):
    print('---save_camera_infomation---')
    rgb_data=camera.get_rgba()[:, :, :3]
    image1 = Image.fromarray(rgb_data, 'RGB')
    save_path=os.path.join(path,f'camera_{step}.png')
    image1.save(save_path)


def save_depth_camera_data(path, camera,step):
    # Depth 데이터 가져오기
    depth_data = camera.get_depth()

    # 정규화에 사용된 최소값과 최대값
    min_value = 0
    max_value = 1250

    # 원래 값으로 복원
    original_depth = depth_data * (max_value - min_value) + min_value

    # PNG 파일 경로 생성
    output_png_path = os.path.join(path, f'depth_{step}.png')

    # # 경로 디버깅
    # print(f"Path: {path}")
    # print(f"Output PNG Path: {output_png_path}")

    # 복원된 Depth 데이터를 uint16 형식으로 저장
    Image.fromarray(original_depth.astype(np.uint16)).save(output_png_path)
    # print(f"복원된 Depth 데이터가 저장되었습니다: {output_png_path}")

def get_camera_bbox_seg_info(camera):
    ############bbox 저장 ###########
    bbox_data=camera.get_current_frame()['bounding_box_2d_loose']
    #bbox_data['data'] = bbox_data['data']
    if isinstance(bbox_data['data'], np.ndarray):
        bbox_data['data'] = bbox_data['data'].tolist()
    #bbox_data['info']['bboxIds'] = bbox_data['info']['bboxIds']


    ################################### segmentation 저장
    seg_data=camera.get_current_frame()['instance_id_segmentation']['data']
    seg_info=camera.get_current_frame()['instance_id_segmentation']['info']
    bbox_data['Segmentation_info'] = seg_info

    # D3_bbox_data=camera.get_current_frame()['bounding_box_3d']
    # #bbox_data['data'] = D3_bbox_data['data']
    # if isinstance(D3_bbox_data['data'], np.ndarray):
    #     D3_bbox_data['data'] = D3_bbox_data['data'].tolist()
    #D3_bbox_data['info']['bboxIds'] = D3_bbox_data['info']['bboxIds'].tolist()





    camera_bbox_data=bbox_data
    camera_segmentation_data=seg_data
    return camera_bbox_data, camera_segmentation_data #D3_bbox_data
