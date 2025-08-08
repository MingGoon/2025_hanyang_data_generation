import random
import numpy as np
import json
from omni.isaac.core import World
from omni.isaac.core.utils.rotations import euler_angles_to_quat 
from omni.isaac.core.utils.prims import get_prim_at_path, delete_prim
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.kit.commands
import omni.usd
from omni.isaac.core.utils.bounds import compute_combined_aabb
from omni.isaac.debug_draw import _debug_draw

try:
    import omni.replicator.core as rep
    REPLICATOR_AVAILABLE = True
except ImportError:
    print("경고: omni.replicator.core 모듈을 찾을 수 없습니다. Annotator 설정이 제한될 수 있습니다.")
    REPLICATOR_AVAILABLE = False
except Exception as e:
    print(f"omni.replicator.core 임포트 중 기타 오류: {e}")
    REPLICATOR_AVAILABLE = False

from omni.isaac.core.prims import XFormPrim, GeometryPrim
from omni.isaac.sensor import Camera
import sys, os
sys.path.append('/home/robot/isaac-sim-4.0/extension_examples')
from omni.isaac.core.utils import viewports, stage, extensions, prims, rotations
from collections import defaultdict
import argparse
import signal
import time
