# file:D:\code\EyeTracking_total\GazeTracking\core\gaze_tracking.py
from __future__ import division

import logging
import math
import os
import cv2
import dlib
import datetime
import time  # 确保导入 time 模块
import math  # 添加这一行

import numpy as np

from core import gaze_direction
from core.dual_3d_kalman_filter import DualDecoupledKalmanFilter
from core.eye import Eye
from core.calibration_direction import DirectionCalibrator, Calibration
from core.gaze_direction import GazeDirectionDetector
from util.utils import get_screen_resolution
from core.dual_3d_kalman_filter import ScreenCoordinateKalmanFilter


class GazeTracking:
    """
    眼动追踪主类，提供瞳孔定位、方向判断、眨眼检测等功能，
    并新增 process_frame 方法以输出结构化的眼动数据。
    专注于从图像中提取原始的眼部特征数据

    人脸检测和面部特征点提取
    瞳孔定位和基本眼部特征提取
    头部姿态估计（输出原始欧拉角）
    输出原始的眼部数据（瞳孔位置、眼角位置、眼睑位置等）
    提供基本的视线比例计算（hr/vr）
    """

    def __init__(self, direction_calibrator=None):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()
        self.direction_calibrator = direction_calibrator or DirectionCalibrator()
        self._face_detector = dlib.get_frontal_face_detector()
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat")
        self._predictor = dlib.shape_predictor(model_path)
        self.gaze_direction_detector = GazeDirectionDetector()
        self.kf = None  # 卡尔曼滤波器实例
        # 添加专用屏幕坐标滤波器
        screen_w, screen_h = get_screen_resolution()
        self.screen_coord_filter = ScreenCoordinateKalmanFilter(screen_w, screen_h)
        self.last_time = None  # 用于计算时间差
        self.frame_count = 0  # 用于控制日志输出频率
        # 添加滤波相关变量
        self.hr_history = []
        self.vr_history = []
        self.filter_window = 5  # 滤波窗口大小

        # 3D人脸模型点 (单位: mm)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),  # 鼻尖
            (-225.0, -170.0, 135.0),  # 左眼外眼角（注意 Y 为负）
            (225.0, -170.0, 135.0),  # 右眼外眼角
            (-150.0, 150.0, 125.0),  # 左嘴角（Y 为正）
            (150.0, 150.0, 125.0),  # 右嘴角
            (0.0, 330.0, 65.0)  # 下巴中心（Y 为正）
        ])

        # 对应的dlib关键点索引
        self.model_points_indices = [30, 36, 45, 48, 54, 8]  # 根据68点模型

        # 相机参数 (假设值，需要根据实际摄像头校准)
        # self.camera_matrix = None
        # self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        # 加载相机校准参数
        self._load_camera_calibration()

        # 添加滤波相关变量
        self.hr_history = []
        self.vr_history = []
        self.filter_window = 5  # 滤波窗口大小

        # 新增：加载用户校准参数（从文件或配置中读取）
        self.user_calib_params = self._load_user_calibration()

        # 新增：动态校准相关属性
        self.error_history = []  # 存储误差历史
        self.dynamic_calibration_triggered = False  # 是否已触发动态校准
        self.dynamic_calibration_point = None  # 动态校准点坐标
        self.dynamic_calibration_samples = []  # 动态校准样本
        self.DYNAMIC_CALIBRATION_THRESHOLD = 50  # 误差阈值（像素）
        self.DYNAMIC_CALIBRATION_FRAMES = 5  # 连续帧数阈值
        self.DYNAMIC_CALIBRATION_SAMPLES_NEEDED = 8  # 需要的样本数

        # 新增：滑动平均滤波相关属性
        self.filtered_coords_history = []  # 存储滤波后的坐标历史
        self.sliding_window_size = 3  # 滑动窗口大小

    def _load_user_calibration(self):
        """加载用户校准参数，包含水平/垂直偏移补偿和敏感度系数"""
        calib_path = os.path.join(os.path.dirname(__file__), "calibration/user_calib.npz")
        if os.path.exists(calib_path):
            try:
                data = np.load(calib_path)
                return {
                    "horizontal_offset": data.get("horizontal_offset", 0.0),  # 水平偏移补偿
                    "vertical_offset": data.get("vertical_offset", 0.0),  # 垂直偏移补偿
                    "horizontal_sens": data.get("horizontal_sens", 1.0),  # 水平敏感度系数
                    "vertical_sens": data.get("vertical_sens", 1.0)  # 垂直敏感度系数
                }
            except Exception as e:
                print(f"[CALIB ERROR] 加载校准参数失败: {e}")
        # 默认参数
        return {"horizontal_offset": 0.0, "vertical_offset": 0.0, "horizontal_sens": 1.0, "vertical_sens": 1.0}

    def _load_camera_calibration(self):
        """加载相机校准参数"""
        calib_path = os.path.join(os.path.dirname(__file__), "calibration/camera_calibration.npz")
        if os.path.exists(calib_path):
            try:
                data = np.load(calib_path)
                self.camera_matrix = data['camera_matrix']
                self.dist_coeffs = data['dist_coeffs']
                return True
            except Exception as e:
                print(f"[CALIB ERROR] 加载相机校准参数失败: {e}")

        # 使用默认值
        self.camera_matrix = np.array([
            [1200, 0, 640],
            [0, 1200, 360],
            [0, 0, 1]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        return False

    # 在 GazeTracking._estimate_head_pose 方法中找到相关代码并进行如下修改：
    def _estimate_head_pose(self, landmarks):
        """估计头部姿态 (旋转和平移向量)"""
        if not landmarks:
            return None, None, None, None

        # 提取2D图像点
        image_points = np.array([
            [landmarks.part(i).x, landmarks.part(i).y]
            for i in self.model_points_indices
        ], dtype=np.float64)

        # 使用校准后的相机矩阵（替换为预定义的相机矩阵）
        # self.camera_matrix = np.array([
        #     [1200, 0, 640],  # 真实焦距1200，图像中心(640,360)
        #     [0, 1200, 360],
        #     [0, 0, 1]
        # ], dtype=np.float64)
        # 确保相机参数已加载
        if self.camera_matrix is None:
            self._load_camera_calibration()

        # 使用solvePnP计算旋转和平移向量
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        # 调试，绘制坐标系，在solvePnP后添加 (成功时)
        if success:
            # 绘制坐标轴 (长度100)
            axis_points = np.float32([[50, 0, 0], [0, 50, 0], [0, 0, 50], [0, 0, 0]])
            img_points, _ = cv2.projectPoints(
                axis_points,
                rotation_vector,
                translation_vector,
                self.camera_matrix,
                self.dist_coeffs
            )

            origin = tuple(img_points[3].ravel().astype(int))
            cv2.line(self.frame, origin, tuple(img_points[0].ravel().astype(int)), (0, 0, 255), 3)  # X轴 (红)
            cv2.line(self.frame, origin, tuple(img_points[1].ravel().astype(int)), (0, 255, 0), 3)  # Y轴 (绿)
            cv2.line(self.frame, origin, tuple(img_points[2].ravel().astype(int)), (255, 0, 0), 3)  # Z轴 (蓝)

        if not success:
            return None, None, None, None

        # 将旋转向量转换为旋转矩阵
        rotation_mat, _ = cv2.Rodrigues(rotation_vector)

        # 将旋转矩阵转换为欧拉角
        pitch, yaw, roll = self._rotation_matrix_to_euler_angles(rotation_mat)

        # 归一化角度
        pitch, yaw, roll = self._normalize_angles(pitch, yaw, roll)

        # 修复头部姿态异常值
        # Roll角异常值处理
        if np.isnan(roll) or abs(roll) > 10:
            if hasattr(self, 'roll_history') and len(self.roll_history) > 0:
                roll = np.mean(self.roll_history)
            else:
                roll = 0.0

        # Yaw角异常值处理
        if np.isnan(yaw) or abs(yaw) > 45:  # 正常yaw范围45度
            if hasattr(self, 'yaw_history') and len(self.yaw_history) > 0:
                yaw = np.mean(self.yaw_history)
            else:
                yaw = 0.0

        # Pitch角异常值处理
        if np.isnan(pitch) or abs(pitch) > 45:  # 正常pitch范围45度
            if hasattr(self, 'pitch_history') and len(self.pitch_history) > 0:
                pitch = np.mean(self.pitch_history)
            else:
                pitch = 0.0

        # Yaw角滑动窗口滤波
        if not hasattr(self, 'yaw_history'):
            self.yaw_history = []
        self.yaw_history.append(yaw)
        if len(self.yaw_history) > 5:
            self.yaw_history.pop(0)
        if len(self.yaw_history) >= 3:
            weights = np.linspace(0.05, 1.0, len(self.yaw_history))
            weights /= np.sum(weights)
            yaw = np.average(self.yaw_history, weights=weights)

        # Pitch角滑动窗口滤波
        if not hasattr(self, 'pitch_history'):
            self.pitch_history = []
        self.pitch_history.append(pitch)
        if len(self.pitch_history) > 5:
            self.pitch_history.pop(0)
        if len(self.pitch_history) >= 3:
            weights = np.linspace(0.05, 1.0, len(self.pitch_history))
            weights /= np.sum(weights)
            pitch = np.average(self.pitch_history, weights=weights)

        # Roll角滑动窗口滤波
        if not hasattr(self, 'roll_history'):
            self.roll_history = []
        self.roll_history.append(roll)
        if len(self.roll_history) > 5:
            self.roll_history.pop(0)
        if len(self.roll_history) >= 3:
            weights = np.linspace(0.05, 1.0, len(self.roll_history))
            weights /= np.sum(weights)
            roll = np.average(self.roll_history, weights=weights)

        return rotation_vector, translation_vector, (pitch, yaw, roll), image_points

    def _rotation_matrix_to_euler_angles(self, R):
        """将旋转矩阵转换为欧拉角 (pitch, yaw, roll)"""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            yaw = np.arctan2(R[1, 0], R[0, 0])
            pitch = np.arctan2(-R[2, 0], sy)
            roll = np.arctan2(R[2, 1], R[2, 2])
        else:
            yaw = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            roll = 0

        return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)

    def _orthonormalize_rotation(self, R):
        """正交化旋转矩阵"""
        U, _, Vt = np.linalg.svd(R)
        return U @ Vt

    def _normalize_angles(self, pitch, yaw, roll):
        """将角度归一化到 [-180, 180] 范围内"""
        pitch = (pitch + 180) % 360 - 180
        yaw = (yaw + 180) % 360 - 180
        roll = (roll + 180) % 360 - 180
        return pitch, yaw, roll

    @property
    def pupils_located(self):
        """检查瞳孔是否已成功定位"""
        try:
            # 检查眼睛对象是否存在
            if self.eye_left is None or self.eye_right is None:
                return False

            # 检查瞳孔对象是否存在
            if self.eye_left.pupil is None or self.eye_right.pupil is None:
                return False

            # 检查瞳孔坐标是否有效
            left_valid = (self.eye_left.pupil.center_x is not None and
                          self.eye_left.pupil.center_y is not None and
                          self.eye_left.pupil.x is not None and
                          self.eye_left.pupil.y is not None)
            right_valid = (self.eye_right.pupil.center_x is not None and
                           self.eye_right.pupil.center_y is not None and
                           self.eye_right.pupil.x is not None and
                           self.eye_right.pupil.y is not None)

            # 至少一只眼睛有有效坐标
            return left_valid or right_valid
        except (TypeError, ValueError, AttributeError):
            return False

    def _analyze(self):
        """检测人脸并初始化左右眼对象"""
        if self.frame is None:
            return

        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(frame)

        # 输出基础检测状态（每帧必输）
        if self.frame_count % 5 == 0:  # 控制输出频率
            print(
                f"[FRAME {self.frame_count}] Faces detected: {len(faces)} | Landmarks detected: {self.landmarks.num_parts if hasattr(self, 'landmarks') and self.landmarks else 0}")

        if len(faces) == 0:
            self.eye_left = None
            self.eye_right = None
            self.landmarks = None
            return

        try:
            landmarks = self._predictor(frame, faces[0])

            # 尝试创建眼部对象
            left_eye_status = "成功"
            right_eye_status = "成功"
            left_error = ""
            right_error = ""

            try:
                self.eye_left = Eye(frame, landmarks, 0, self.calibration)
                self.eye_left.gaze_tracking = self  # 传递GazeTracking实例到左眼
            except Exception as e:
                self.eye_left = None
                left_eye_status = "失败"
                left_error = str(e)

            try:
                self.eye_right = Eye(frame, landmarks, 1, self.calibration)
                self.eye_right.gaze_tracking = self  # 传递GazeTracking实例到右眼
            except Exception as e:
                self.eye_right = None
                right_eye_status = "失败"
                right_error = str(e)

            # 输出眼部状态（每帧必输）
            if self.frame_count % 5 == 0:  # 控制输出频率
                if left_eye_status == "失败" or right_eye_status == "失败":
                    print(f"[EYE DEBUG] 左眼状态: {left_eye_status}" + (f" - {left_error}" if left_error else "") +
                          f" | 右眼状态: {right_eye_status}" + (f" - {right_error}" if right_error else ""))
                else:
                    print(f"[EYE DEBUG] 左眼状态: {left_eye_status} | 右眼状态: {right_eye_status}")

            # 保存特征点检测结果
            self.landmarks = landmarks
        except Exception as e:
            self.eye_left = None
            self.eye_right = None
            self.landmarks = None

    def refresh(self, frame):
        """刷新当前帧并进行分析"""
        self.frame = frame
        self.frame_count += 1

        # 初始化双解耦卡尔曼滤波器（如果尚未初始化）
        if self.kf is None:
            screen_w, screen_h = get_screen_resolution()
            dt = 1.0 / 30.0  # 假设30fps
            self.kf = DualDecoupledKalmanFilter(dt, screen_w, screen_h)
            self.last_time = time.time()

        self._analyze()

    def pupil_left_coords(self):
        """获取左眼瞳孔坐标"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.center_x
            y = self.eye_left.origin[1] + self.eye_left.pupil.center_y
            return (x, y)
        return None

    def pupil_right_coords(self):
        """获取右眼瞳孔坐标"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.center_x
            y = self.eye_right.origin[1] + self.eye_right.pupil.center_y
            return (x, y)
        return None

    def horizontal_ratio(self):
        """
        计算水平注视比例 (0-1)，0表示最左侧，1表示最右侧
        增加单眼数据降级处理和头部姿态修正
        """
        avg_ratio = None  # 初始化为None
        if not self.pupils_located:
            return None

        try:
            # 1. 尝试获取双眼瞳孔原始坐标
            left_coords = self.eye_left.pupil_coords() if self.eye_left else None
            right_coords = self.eye_right.pupil_coords() if self.eye_right else None

            # 2. 单眼降级逻辑
            if left_coords and not right_coords:
                # 仅左眼有效：基于左眼眼角计算
                if hasattr(self, 'landmarks') and self.landmarks:
                    left_inner_x = self.landmarks.part(39).x
                    left_outer_x = self.landmarks.part(36).x
                    left_width = abs(left_outer_x - left_inner_x)
                    if left_width == 0:
                        return 0.5
                    left_min = min(left_inner_x, left_outer_x)
                    ratio = (left_coords[0] - left_min) / left_width
                    avg_ratio = max(0.0, min(1.0, ratio))

            elif right_coords and not left_coords:
                # 仅右眼有效：基于右眼眼角计算
                if hasattr(self, 'landmarks') and self.landmarks:
                    right_inner_x = self.landmarks.part(42).x
                    right_outer_x = self.landmarks.part(45).x
                    right_width = abs(right_outer_x - right_inner_x)
                    if right_width == 0:
                        return 0.5
                    right_min = min(right_inner_x, right_outer_x)
                    ratio = (right_coords[0] - right_min) / right_width
                    avg_ratio = max(0.0, min(1.0, ratio))

            elif left_coords and right_coords:
                # 双眼有效：使用原有逻辑计算
                if (hasattr(self, 'landmarks') and self.landmarks and
                        hasattr(self.eye_left, 'origin') and hasattr(self.eye_right, 'origin')):

                    # 获取左眼参数
                    left_inner_x = self.landmarks.part(39).x
                    left_outer_x = self.landmarks.part(36).x
                    left_width = abs(left_outer_x - left_inner_x)

                    # 计算左眼相对位置比例
                    if left_width > 0:
                        left_min = min(left_inner_x, left_outer_x)
                        left_ratio = (left_coords[0] - left_min) / left_width
                    else:
                        left_ratio = 0.5

                    # 获取右眼参数
                    right_inner_x = self.landmarks.part(42).x
                    right_outer_x = self.landmarks.part(45).x
                    right_width = abs(right_outer_x - right_inner_x)

                    # 计算右眼相对位置比例
                    if right_width > 0:
                        right_min = min(right_inner_x, right_outer_x)
                        right_ratio = (right_coords[0] - right_min) / right_width
                    else:
                        right_ratio = 0.5

                    # 返回双眼平均值
                    avg_ratio = (left_ratio + right_ratio) / 2
                    avg_ratio = max(0.0, min(1.0, avg_ratio))
            else:
                return 0.5

            # 头部姿态补偿逻辑
            if avg_ratio is not None and hasattr(self, 'landmarks'):
                # 获取头部姿态
                _, _, euler_angles, _ = self._estimate_head_pose(self.landmarks)
                if euler_angles:
                    pitch, yaw, roll = euler_angles
                    screen_w, screen_h = get_screen_resolution()

                    # 应用头部偏航角补偿
                    original_hr = avg_ratio  # 保存原始HR值
                    if hasattr(self.gaze_direction_detector, 'compensate_head_yaw'):
                        # 获取目标点x坐标（如果可用）
                        target_x = None
                        if hasattr(self, 'gaze_data') and self.gaze_data.get("target_coords"):
                            target_x = self.gaze_data["target_coords"][0]

                        avg_ratio = self.gaze_direction_detector.compensate_head_yaw(
                            avg_ratio, yaw, target_x
                        )

                        # 添加补偿监控日志
                        # 添加补偿监控日志
                        print(
                            f"[补偿监控] Yaw: {yaw:.2f}°，原始hr: {original_hr:.3f}，补偿后hr: {avg_ratio:.3f}，补偿量: {avg_ratio - original_hr:.3f}")
                    else:
                        # 如果没有专门的补偿方法，使用原有的补偿逻辑
                        # 1. 获取真实目标坐标（校准阶段）
                        if hasattr(self, 'gaze_data') and self.gaze_data.get("target_coords"):
                            target_x = self.gaze_data["target_coords"][0]
                        else:
                            target_x = screen_w // 2

                        # 2. 方向判断（目标在左，头部左转时应增强补偿）
                        target_is_left = target_x < screen_w // 2  # 如calibration_point_1在左
                        head_turn_left = yaw < 0  # Yaw负=头部左转
                        compensation_weight = 0.12 if (target_is_left and head_turn_left) else 0.03

                        # 3. 修正补偿符号：头部左转减小HR使screen_x增大（向目标161靠近）
                        if head_turn_left:
                            avg_ratio -= abs(yaw) * compensation_weight  # 减小HR
                        elif yaw > 0:  # 头部右转
                            avg_ratio += yaw * compensation_weight  # 增大HR

                        # 添加补偿监控日志
                        compensation_amount = avg_ratio - original_hr
                        print(f"[补偿监控] Yaw: {yaw:.2f}°，原始hr: {original_hr:.3f}，补偿后hr: {avg_ratio:.3f}，补偿量: {compensation_amount:.3f}")

                    avg_ratio = max(0.0, min(1.0, avg_ratio))

            return avg_ratio

        except Exception as e:
            print(f"[GAZE ERROR] 计算水平比例失败: {str(e)}")
            return 0.5

    # 修改 vertical_ratio 方法以使用 get_left_eye_center_y
    def vertical_ratio(self):
        """
        计算垂直注视比例 (0-1)，0表示最上方，1表示最下方
        修复版：确保不返回 None，增加滑动窗口加权滤波
        """
        if not self.pupils_located:
            return None

        try:
            # 检查 pupil.y 是否存在
            if (self.eye_left is None or self.eye_right is None or
                    self.eye_left.pupil is None or self.eye_right.pupil is None or
                    self.eye_left.pupil.y is None or self.eye_right.pupil.y is None):
                return 0.5

            # 优先使用眼球中心和眼睑坐标计算
            if (hasattr(self.eye_left, 'get_left_eye_center_y') and
                    hasattr(self.eye_right, 'eye_center_y') and
                    hasattr(self.eye_left, 'upper_lid_y') and hasattr(self.eye_left, 'lower_lid_y') and
                    hasattr(self.eye_right, 'upper_lid_y') and hasattr(self.eye_right, 'lower_lid_y')):

                # 获取左眼中心Y坐标（使用增强版方法）
                left_eye_center_y = self.eye_left.get_left_eye_center_y()
                # 获取右眼中心Y坐标
                right_eye_center_y = self.eye_right.eye_center_y

                # 获取眼睑坐标
                left_upper_y = self.eye_left.upper_lid_y
                left_lower_y = self.eye_left.lower_lid_y
                right_upper_y = self.eye_right.upper_lid_y
                right_lower_y = self.eye_right.lower_lid_y

                # 计算眼睑间距
                left_height = left_lower_y - left_upper_y
                right_height = right_lower_y - right_upper_y

                # 计算瞳孔垂直位置（相对于眼球中心）
                left_pupil_y = self.eye_left.origin[1] + self.eye_left.pupil.y
                right_pupil_y = self.eye_right.origin[1] + self.eye_right.pupil.y

                # 计算相对位置比例
                if left_height > 0:
                    left_ratio = (left_pupil_y - left_eye_center_y) / left_height
                else:
                    left_ratio = 0.0

                if right_height > 0:
                    right_ratio = (right_pupil_y - right_eye_center_y) / right_height
                else:
                    right_ratio = 0.0

                # 返回双眼平均值
                avg_ratio = (left_ratio + right_ratio) / 2

            else:
                # 备选方案：使用眼睛ROI高度
                left_eye_height = self.eye_left.frame.shape[0] if (
                        self.eye_left and self.eye_left.frame is not None) else 1
                right_eye_height = self.eye_right.frame.shape[0] if (
                        self.eye_right and self.eye_right.frame is not None) else 1

                # 计算左眼比例
                if left_eye_height > 0 and self.eye_left.pupil:
                    left_ratio = self.eye_left.pupil.y / left_eye_height
                else:
                    left_ratio = 0.5

                # 计算右眼比例
                if right_eye_height > 0 and self.eye_right.pupil:
                    right_ratio = self.eye_right.pupil.y / right_eye_height
                else:
                    right_ratio = 0.5

                # 计算平均比例
                avg_ratio = (left_ratio + right_ratio) / 2

            # 新增：滑动窗口加权滤波（窗口大小=5，权重递减，增强最新帧权重）
            if avg_ratio is not None:
                # 确保vr_history属性存在
                if not hasattr(self, 'vr_history'):
                    self.vr_history = []

                self.vr_history.append(avg_ratio)
                if len(self.vr_history) > 5:
                    self.vr_history.pop(0)
                # 加权滤波：最新帧权重0.4，前1帧0.3，前2帧0.2，前3帧0.1，前4帧0.0（仅当窗口满5帧）
                if len(self.vr_history) == 5:
                    weights = [0.0, 0.1, 0.2, 0.3, 0.4]
                    avg_ratio = sum(w * r for w, r in zip(weights, self.vr_history))
                elif len(self.vr_history) > 1:
                    # 如果窗口未满但有数据，使用简单平均
                    avg_ratio = sum(self.vr_history) / len(self.vr_history)

            clamped_ratio = max(0.0, min(1.0, avg_ratio))
            return clamped_ratio

        except (ZeroDivisionError, AttributeError, TypeError):
            return 0.5

    def is_right(self):
        return self.horizontal_ratio() is not None and self.horizontal_ratio() <= 0.35

    def is_left(self):
        return self.horizontal_ratio() is not None and self.horizontal_ratio() >= 0.65

    def is_center(self):
        return self.is_right() is not True and self.is_left() is not True

    def is_up(self):
        ratio = self.vertical_ratio()
        return ratio is not None and ratio < 0.35

    def is_down(self):
        ratio = self.vertical_ratio()
        return ratio is not None and ratio > 0.65

    def is_blinking(self):
        """判断是否眨眼（修复版：综合双眼比率和瞳孔有效性）"""
        # 1. 校验双眼是否存在
        if self.eye_left is None or self.eye_right is None:
            return False

        # 2. 提取双眼眨眼比率（处理 None 情况）
        left_ratio = self.eye_left.blinking if self.eye_left.blinking is not None else 0.0
        right_ratio = self.eye_right.blinking if self.eye_right.blinking is not None else 0.0

        # 3. 提取双眼瞳孔有效性
        left_valid = self.eye_left.pupil.is_valid if (self.eye_left.pupil) else False
        right_valid = self.eye_right.pupil.is_valid if (self.eye_right.pupil) else False

        # 4. 判定逻辑：双眼比率均>5.5 或 双眼瞳孔均无效
        ratio_blink = (left_ratio > 5.5) and (right_ratio > 5.5)
        pupil_invalid = not (left_valid or right_valid)
        is_blink = ratio_blink or pupil_invalid

        return is_blink

    def annotated_frame(self):
        """返回带有标注信息的图像帧"""
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()

            cv2.circle(frame, (x_left, y_left), radius=1, color=color, thickness=-1)
            cv2.circle(frame, (x_right, y_right), radius=1, color=color, thickness=-1)

            if self.eye_left.pupil and self.eye_left.pupil.is_valid:
                cv2.circle(frame, (x_left, y_left), int(self.eye_left.pupil.radius), color, 1)
            if self.eye_right.pupil and self.eye_right.pupil.is_valid:
                cv2.circle(frame, (x_right, y_right), int(self.eye_right.pupil.radius), color, 1)

        return frame

    def extract_features_from_circle(self, landmarks, pupil_circle, head_pose, eye_side='right'):
        """
        基于圆拟合结果提取特征

        参数:
            landmarks: 眼部关键点坐标
            pupil_circle: 瞳孔圆信息 (center_x, center_y, radius)
            head_pose: 头部姿态 (pitch, yaw, roll)
            eye_side: 眼睛类型 ('left' 或 'right')，默认为 'right'

        返回:
            features: 特征字典

        """
        # 根据眼睛类型确定关键点索引
        if eye_side == 'left':
            # 左眼: 内眼角是 landmarks[3]，外眼角是 landmarks[0]
            inner_corner = landmarks[3]  # 内眼角
            outer_corner = landmarks[0]  # 外眼角
        else:  # 默认右眼
            # 右眼: 内眼角是 landmarks[0]，外眼角是 landmarks[3]
            inner_corner = landmarks[0]  # 内眼角
            outer_corner = landmarks[3]  # 外眼角

        # 计算眼部坐标系原点和宽度
        eye_origin = ((inner_corner[0] + outer_corner[0]) / 2,
                      (inner_corner[1] + outer_corner[1]) / 2)
        eye_width = math.sqrt((inner_corner[0] - outer_corner[0]) ** 2 + (inner_corner[1] - outer_corner[1]) ** 2)

        # 计算归一化偏移量
        pupil_center = (pupil_circle[0], pupil_circle[1])
        delta_x = pupil_center[0] - eye_origin[0]
        delta_y = pupil_center[1] - eye_origin[1]
        norm_dx = delta_x / eye_width if eye_width > 0 else 0
        norm_dy = delta_y / eye_width if eye_width > 0 else 0

        # 计算归一化瞳孔大小
        norm_radius = pupil_circle[2] / eye_width if eye_width > 0 else 0

        # 头部姿态特征
        pitch, yaw, roll = head_pose[:3]

        # 构建特征向量
        features = {
            'norm_dx': norm_dx,
            'norm_dy': norm_dy,
            'norm_radius': norm_radius,
            'head_pitch': pitch,
            'head_yaw': yaw,
            'head_roll': roll,
            'eye_side': eye_side  # 添加眼睛类型标识
        }

        return features

    # 在 GazeTracking.process_frame 方法中找到相关代码并进行如下修改：

    def process_frame(self):
        """
        处理当前帧，返回结构化的注视数据
        包含时间戳、瞳孔坐标、视线比例、方向、屏幕坐标等信息
        """
        # 初始化 gaze_data 字典
        gaze_data = self._initialize_gaze_data()

        # 1.提取眼部特征
        self._extract_eye_features(gaze_data)

        # 2.处理头部姿态
        self._process_head_pose(gaze_data)

        # 3.计算视线比例
        self._calculate_gaze_ratios(gaze_data)

        # 4.计算屏幕坐标
        self._calculate_screen_coordinates(gaze_data)

        # 5.应用滤波处理
        self._apply_filters(gaze_data)

        # 6.处理3D注视向量
        self._process_3d_gaze_vector(gaze_data)

        # 7.整合生理特征
        self._integrate_physiological_features(gaze_data)

        # 8.输出调试信息
        self._output_debug_info(gaze_data)

        # 9.处理动态校准
        self._handle_dynamic_calibration(gaze_data)

        return gaze_data

    def _initialize_gaze_data(self):
        """初始化 gaze_data 字典"""
        return {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "left_pupil": self.pupil_left_coords(),
            "right_pupil": self.pupil_right_coords(),
            "left_pupil_x": None,
            "left_pupil_y": None,
            "right_pupil_x": None,
            "right_pupil_y": None,
            "left_pupil_radius": self.eye_left.pupil.radius if self.eye_left and self.eye_left.pupil else None,
            "right_pupil_radius": self.eye_right.pupil.radius if self.eye_right and self.eye_right.pupil else None,
            # 眼角坐标
            "left_eye_inner_x": None,
            "left_eye_inner_y": None,
            "left_eye_outer_x": None,
            "left_eye_outer_y": None,
            "right_eye_inner_x": None,
            "right_eye_inner_y": None,
            "right_eye_outer_x": None,
            "right_eye_outer_y": None,
            # 上下眼睑坐标
            "left_eye_upper_lid": None,
            "left_eye_lower_lid": None,
            "right_eye_upper_lid": None,
            "right_eye_lower_lid": None,
            "hr": None,
            "vr": None,
            "direction": None,
            "is_blinking": self.is_blinking(),
            # 屏幕坐标相关
            "screen_coords": None,
            "target_coords": None,
            "head_pose": None,
            "screen_coords_head": None,
        }

    def _extract_eye_features(self, gaze_data):
        """提取眼部特征信息"""
        # 获取最新的左右瞳孔x,y坐标
        left_coords = self.pupil_left_coords()
        if left_coords:
            gaze_data["left_pupil_x"], gaze_data["left_pupil_y"] = left_coords

        right_coords = self.pupil_right_coords()
        if right_coords:
            gaze_data["right_pupil_x"], gaze_data["right_pupil_y"] = right_coords

        # 计算瞳孔半径变化率
        self._calculate_pupil_radius_change(gaze_data)

        # 眼角坐标计算
        self._calculate_eye_corner_coordinates(gaze_data)

        # 从 Eye 对象获取眼球中心 Y 坐标和上下眼睑 Y 坐标
        if self.eye_left and hasattr(self.eye_left, 'eye_center_y'):
            gaze_data["left_eye_center_y"] = self.eye_left.eye_center_y
            gaze_data["left_eye_upper_lid_y"] = self.eye_left.upper_lid_y
            gaze_data["left_eye_lower_lid_y"] = self.eye_left.lower_lid_y
        if self.eye_right and hasattr(self.eye_right, 'eye_center_y'):
            gaze_data["right_eye_center_y"] = self.eye_right.eye_center_y
            gaze_data["right_eye_upper_lid_y"] = self.eye_right.upper_lid_y
            gaze_data["right_eye_lower_lid_y"] = self.eye_right.lower_lid_y

    def _calculate_pupil_radius_change(self, gaze_data):
        """计算瞳孔半径变化率"""
        if self.eye_left and self.eye_left.pupil and self.eye_left.pupil.radius:
            left_radius = self.eye_left.pupil.radius
            # 保存历史半径值用于计算变化率
            if not hasattr(self, 'left_pupil_radius_history'):
                self.left_pupil_radius_history = []
            self.left_pupil_radius_history.append(left_radius)
            if len(self.left_pupil_radius_history) > 4:  # 保留最近4帧数据
                self.left_pupil_radius_history.pop(0)

            # 计算与前3帧的平均半径变化
            if len(self.left_pupil_radius_history) >= 4:
                avg_prev_radius = np.mean(self.left_pupil_radius_history[:-1])
                gaze_data["left_pupil_radius_change"] = left_radius - avg_prev_radius
            else:
                gaze_data["left_pupil_radius_change"] = 0

        if self.eye_right and self.eye_right.pupil and self.eye_right.pupil.radius:
            right_radius = self.eye_right.pupil.radius
            # 保存历史半径值用于计算变化率
            if not hasattr(self, 'right_pupil_radius_history'):
                self.right_pupil_radius_history = []
            self.right_pupil_radius_history.append(right_radius)
            if len(self.right_pupil_radius_history) > 4:  # 保留最近4帧数据
                self.right_pupil_radius_history.pop(0)

            # 计算与前3帧的平均半径变化
            if len(self.right_pupil_radius_history) >= 4:
                avg_prev_radius = np.mean(self.right_pupil_radius_history[:-1])
                gaze_data["right_pupil_radius_change"] = right_radius - avg_prev_radius
            else:
                gaze_data["right_pupil_radius_change"] = 0

        # 设置主视眼的瞳孔半径变化率
        if hasattr(self.direction_calibrator, 'primary_eye') and self.direction_calibrator.primary_eye == 'left':
            gaze_data["pupil_radius_change"] = gaze_data.get("left_pupil_radius_change", 0)
        else:
            gaze_data["pupil_radius_change"] = gaze_data.get("right_pupil_radius_change", 0)

    def _calculate_eye_corner_coordinates(self, gaze_data):
        """计算眼角坐标"""
        try:
            # 优先使用已保存的特征点
            if self.landmarks:
                # 眼角坐标
                left_eye_inner = (self.landmarks.part(39).x, self.landmarks.part(39).y)
                left_eye_outer = (self.landmarks.part(36).x, self.landmarks.part(36).y)
                right_eye_inner = (self.landmarks.part(42).x, self.landmarks.part(42).y)
                right_eye_outer = (self.landmarks.part(45).x, self.landmarks.part(45).y)

                gaze_data["left_eye_inner"] = left_eye_inner
                gaze_data["left_eye_outer"] = left_eye_outer
                gaze_data["right_eye_inner"] = right_eye_inner
                gaze_data["right_eye_outer"] = right_eye_outer

                # 上下眼睑坐标
                left_eye_upper_lid = (
                    (self.landmarks.part(37).x + self.landmarks.part(38).x) / 2,
                    (self.landmarks.part(37).y + self.landmarks.part(38).y) / 2
                )
                left_eye_lower_lid = (
                    (self.landmarks.part(40).x + self.landmarks.part(41).x) / 2,
                    (self.landmarks.part(40).y + self.landmarks.part(41).y) / 2
                )
                right_eye_upper_lid = (
                    (self.landmarks.part(43).x + self.landmarks.part(44).x) / 2,
                    (self.landmarks.part(43).y + self.landmarks.part(44).y) / 2
                )
                right_eye_lower_lid = (
                    (self.landmarks.part(46).x + self.landmarks.part(47).x) / 2,
                    (self.landmarks.part(46).y + self.landmarks.part(47).y) / 2
                )

                gaze_data["left_eye_upper_lid"] = left_eye_upper_lid
                gaze_data["left_eye_lower_lid"] = left_eye_lower_lid
                gaze_data["right_eye_upper_lid"] = right_eye_upper_lid
                gaze_data["right_eye_lower_lid"] = right_eye_lower_lid

                # 确保所有需要的字段都被设置
                gaze_data["left_eye_inner_x"] = left_eye_inner[0]
                gaze_data["left_eye_inner_y"] = left_eye_inner[1]
                gaze_data["left_eye_outer_x"] = left_eye_outer[0]
                gaze_data["left_eye_outer_y"] = left_eye_outer[1]
                gaze_data["right_eye_inner_x"] = right_eye_inner[0]
                gaze_data["right_eye_inner_y"] = right_eye_inner[1]
                gaze_data["right_eye_outer_x"] = right_eye_outer[0]
                gaze_data["right_eye_outer_y"] = right_eye_outer[1]

                gaze_data["left_eye_upper_lid_x"] = left_eye_upper_lid[0]
                gaze_data["left_eye_upper_lid_y"] = left_eye_upper_lid[1]
                gaze_data["left_eye_lower_lid_x"] = left_eye_lower_lid[0]
                gaze_data["left_eye_lower_lid_y"] = left_eye_lower_lid[1]
                gaze_data["right_eye_upper_lid_x"] = right_eye_upper_lid[0]
                gaze_data["right_eye_upper_lid_y"] = right_eye_upper_lid[1]
                gaze_data["right_eye_lower_lid_x"] = right_eye_lower_lid[0]
                gaze_data["right_eye_lower_lid_y"] = right_eye_lower_lid[1]

            else:
                # 如果没有保存的特征点，重新检测
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                faces = self._face_detector(gray)
                if faces:
                    landmarks = self._predictor(gray, faces[0])
                    self.landmarks = landmarks  # 保存新的特征点

                    # 眼角坐标
                    left_eye_inner = (landmarks.part(39).x, landmarks.part(39).y)
                    left_eye_outer = (landmarks.part(36).x, landmarks.part(36).y)
                    right_eye_inner = (landmarks.part(42).x, landmarks.part(42).y)
                    right_eye_outer = (landmarks.part(45).x, landmarks.part(45).y)

                    gaze_data["left_eye_inner"] = left_eye_inner
                    gaze_data["left_eye_outer"] = left_eye_outer
                    gaze_data["right_eye_inner"] = right_eye_inner
                    gaze_data["right_eye_outer"] = right_eye_outer

                    # 上下眼睑坐标
                    left_eye_upper_lid = (
                        (landmarks.part(37).x + landmarks.part(38).x) / 2,
                        (landmarks.part(37).y + landmarks.part(38).y) / 2
                    )
                    left_eye_lower_lid = (
                        (landmarks.part(40).x + landmarks.part(41).x) / 2,
                        (landmarks.part(40).y + landmarks.part(41).y) / 2
                    )
                    right_eye_upper_lid = (
                        (landmarks.part(43).x + landmarks.part(44).x) / 2,
                        (landmarks.part(43).y + landmarks.part(44).y) / 2
                    )
                    right_eye_lower_lid = (
                        (landmarks.part(46).x + landmarks.part(47).x) / 2,
                        (landmarks.part(46).y + landmarks.part(47).y) / 2
                    )

                    gaze_data["left_eye_upper_lid"] = left_eye_upper_lid
                    gaze_data["left_eye_lower_lid"] = left_eye_lower_lid
                    gaze_data["right_eye_upper_lid"] = right_eye_upper_lid
                    gaze_data["right_eye_lower_lid"] = right_eye_lower_lid

                    # 确保所有需要的字段都被设置
                    gaze_data["left_eye_inner_x"] = left_eye_inner[0]
                    gaze_data["left_eye_inner_y"] = left_eye_inner[1]
                    gaze_data["left_eye_outer_x"] = left_eye_outer[0]
                    gaze_data["left_eye_outer_y"] = left_eye_outer[1]
                    gaze_data["right_eye_inner_x"] = right_eye_inner[0]
                    gaze_data["right_eye_inner_y"] = right_eye_inner[1]
                    gaze_data["right_eye_outer_x"] = right_eye_outer[0]
                    gaze_data["right_eye_outer_y"] = right_eye_outer[1]

                    gaze_data["left_eye_upper_lid_x"] = left_eye_upper_lid[0]
                    gaze_data["left_eye_upper_lid_y"] = left_eye_upper_lid[1]
                    gaze_data["left_eye_lower_lid_x"] = left_eye_lower_lid[0]
                    gaze_data["left_eye_lower_lid_y"] = left_eye_lower_lid[1]
                    gaze_data["right_eye_upper_lid_x"] = right_eye_upper_lid[0]
                    gaze_data["right_eye_upper_lid_y"] = right_eye_upper_lid[1]
                    gaze_data["right_eye_lower_lid_x"] = right_eye_lower_lid[0]
                    gaze_data["right_eye_lower_lid_y"] = right_eye_lower_lid[1]
                else:
                    # 设置默认值
                    self._set_default_eye_features(gaze_data)

        except Exception as e:
            # 特征点检测失败时使用默认值
            self._set_default_eye_features(gaze_data)

    def _process_head_pose(self, gaze_data):
        """处理头部姿态估计"""
        if hasattr(self, 'landmarks') and self.landmarks:
            # 获取头部姿态的欧拉角（俯仰角pitch, 偏航角yaw, 翻滚角roll）
            _, _, euler_angles, _ = self._estimate_head_pose(self.landmarks)
            if euler_angles:
                gaze_data["head_pose"] = euler_angles

        # 将 head_pose 拆分为单独字段
        if gaze_data.get("head_pose"):
            pitch, yaw, roll = gaze_data["head_pose"]
            gaze_data["head_pitch"] = pitch
            gaze_data["head_yaw"] = yaw
            gaze_data["head_roll"] = roll
        else:
            gaze_data["head_pitch"] = None
            gaze_data["head_yaw"] = None
            gaze_data["head_roll"] = None

        # 添加滚转校正
        if gaze_data.get("head_pose"):
            # 提取翻滚角
            _, _, roll_angle = gaze_data["head_pose"]
            # 使用 GazeDirectionDetector 进行校正
            gaze_data = self.gaze_direction_detector.apply_roll_correction(
                gaze_data,
                roll_angle
            )

        # 传递用户自定义参数到gaze_data
        if hasattr(self, 'user_calib_params'):
            gaze_data["user_horizontal_sens"] = self.user_calib_params.get("horizontal_sens", 1.0)
            gaze_data["user_vertical_sens"] = self.user_calib_params.get("vertical_sens", 1.0)

    def _calculate_gaze_ratios(self, gaze_data):
        """计算视线比例"""
        gaze_data["hr"] = self.horizontal_ratio()
        gaze_data["vr"] = self.vertical_ratio()

        # 应用移动平均滤波
        self._apply_ratio_filtering(gaze_data)

        # 严格异常值过滤
        if gaze_data["hr"] is not None:
            gaze_data["hr"] = max(0.05, min(0.95, gaze_data["hr"]))
        if gaze_data["vr"] is not None:
            gaze_data["vr"] = max(0.05, min(0.95, gaze_data["vr"]))

    def _apply_ratio_filtering(self, gaze_data):
        """应用比例值滤波"""
        # 水平比例滤波
        if gaze_data["hr"] is not None:
            self.hr_history.append(gaze_data["hr"])
            if len(self.hr_history) > self.filter_window:
                self.hr_history.pop(0)
            # 剔除异常值后再计算平均
            hr_values = np.array(self.hr_history)
            if len(hr_values) > 3:
                # 剔除离群值
                q75, q25 = np.percentile(hr_values, [75, 25])
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                filtered_hr = hr_values[(hr_values >= lower_bound) & (hr_values <= upper_bound)]
                if len(filtered_hr) > 0:
                    gaze_data["hr"] = np.mean(filtered_hr)

        # 垂直比例滤波
        if gaze_data["vr"] is not None:
            self.vr_history.append(gaze_data["vr"])
            if len(self.vr_history) > self.filter_window:
                self.vr_history.pop(0)
            # 剔除异常值后再计算平均
            vr_values = np.array(self.vr_history)
            if len(vr_values) > 3:
                # 剔除离群值
                q75, q25 = np.percentile(vr_values, [75, 25])
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                filtered_vr = vr_values[(vr_values >= lower_bound) & (vr_values <= upper_bound)]
                if len(filtered_vr) > 0:
                    gaze_data["vr"] = np.mean(filtered_vr)

    # 在 gaze_tracking.py 的 process_frame 方法中找到 _calculate_screen_coordinates 调用部分
    def _calculate_screen_coordinates(self, gaze_data):
        """计算屏幕坐标"""
        screen_w, screen_h = get_screen_resolution()
        if gaze_data["hr"] is not None and gaze_data["vr"] is not None:
            # 优先使用高级映射模型
            try:
                use_advanced = (hasattr(self.direction_calibrator, 'predict_screen_coordinates_advanced') and
                                self.direction_calibrator.is_calibrated() and
                                hasattr(self.direction_calibrator, 'calibration_samples') and
                                len(self.direction_calibrator.calibration_samples) >= 20)
                if use_advanced:
                    screen_coords = self.direction_calibrator.predict_screen_coordinates_advanced(gaze_data)
                    # 应用用户校准偏移补偿
                    calib = self.user_calib_params
                    screen_coords = (
                        int(screen_coords[0] + calib["horizontal_offset"] * calib["horizontal_sens"]),
                        int(screen_coords[1] + calib["vertical_offset"] * calib["vertical_sens"])
                    )
                    # 校验坐标连续性
                    if hasattr(self, 'prev_raw_screen_coords') and self.prev_raw_screen_coords:
                        prev_x, prev_y = self.prev_raw_screen_coords
                        curr_x, curr_y = screen_coords
                        max_delta_x = screen_w * 0.1
                        max_delta_y = screen_h * 0.1
                        # 若跳变过大，使用上一帧坐标平滑过渡
                        if abs(curr_x - prev_x) > max_delta_x or abs(curr_y - prev_y) > max_delta_y:
                            screen_coords = (
                                int(prev_x + (curr_x - prev_x) * 0.3),
                                int(prev_y + (curr_y - prev_y) * 0.3)
                            )
                    self.prev_raw_screen_coords = screen_coords
                else:
                    # 使用 GazeDirectionDetector 的 calculate_screen_coords 方法
                    if hasattr(self, 'gaze_direction_detector'):
                        hr = gaze_data.get("hr", 0.5)
                        vr = gaze_data.get("vr", 0.5)
                        head_pose = gaze_data.get("head_pose", (0, 0, 0))
                        if len(head_pose) >= 2:
                            yaw, pitch = head_pose[1], head_pose[0]
                        else:
                            yaw, pitch = 0, 0

                        screen_x, screen_y = self.gaze_direction_detector.calculate_screen_coords(
                            hr, vr, yaw, pitch, screen_w, screen_h
                        )
                        screen_coords = (screen_x, screen_y)
                    else:
                        # 核心修改：HR/VR直接贡献70%坐标，头部姿态补偿贡献30%
                        screen_coords = self._calculate_weighted_screen_coordinates(gaze_data)

                gaze_data["target_coords"] = screen_coords
                gaze_data["screen_coords"] = screen_coords
            except Exception as e:
                # 异常时使用线性映射+范围约束
                base_x = max(20, min(int(gaze_data["hr"] * screen_w), screen_w - 20))
                base_y = max(20, min(int(gaze_data["vr"] * screen_h), screen_h - 20))
                screen_coords = (base_x, base_y)
                gaze_data["target_coords"] = screen_coords
                gaze_data["screen_coords"] = screen_coords
        else:
            # HR/VR无效时，使用上一帧坐标或屏幕中心
            if hasattr(self, 'prev_raw_screen_coords') and self.prev_raw_screen_coords:
                gaze_data["screen_coords"] = self.prev_raw_screen_coords
            else:
                gaze_data["screen_coords"] = (screen_w // 2, screen_h // 2)

        # 严格异常值过滤 - 坐标跳变过滤
        self._filter_coordinate_jumps(gaze_data)

    def _calculate_weighted_screen_coordinates(self, gaze_data):
        """计算加权屏幕坐标"""
        screen_w, screen_h = get_screen_resolution()

        # 修复水平映射 key 不匹配问题
        if gaze_data.get("direction", "").startswith("calibration_point"):
            # 关键：使用example.py传递的"target_coords"（真实校准点坐标）
            real_target = gaze_data.get("target_coords")
            if real_target:
                target_x, target_y = real_target
            else:
                print(f"[ERROR] 未传递真实目标坐标！当前direction: {gaze_data.get('direction')}")
                target_x, target_y = (screen_w // 2, screen_h // 2)
        else:
            target_x, target_y = gaze_data.get("target_coords", (screen_w // 2, screen_h // 2))

        # 重新计算corrected_hr
        target_hr = (target_x / screen_w)
        if target_hr > 0.01 and gaze_data["hr"] > 0.01:
            corrected_hr = target_hr
        else:
            corrected_hr = gaze_data["hr"]

        # 重构X坐标映射：HR权重70% + 头部补偿30%
        base_x = corrected_hr * screen_w * 0.7

        # 头部姿态补偿贡献30%
        if gaze_data.get("screen_coords_head"):
            head_x, head_y = gaze_data["screen_coords_head"]
            base_x += head_x * 0.3

        # 应用用户校准参数
        calib = self.user_calib_params
        base_x += calib["horizontal_offset"] * calib["horizontal_sens"]

        # 微小抖动模拟
        jitter = np.random.normal(0, 2)

        # 重构Y坐标映射
        # 1. 计算VR的理论目标值
        target_vr = (target_y / screen_h)

        # 2. 修正VR
        if target_vr > 0.01 and gaze_data["vr"] > 0.01:
            corrected_vr = gaze_data["vr"] * (target_vr / gaze_data["vr"]) * 0.7 + gaze_data["vr"] * 0.3
        else:
            corrected_vr = gaze_data["vr"]

        # 3. 重构Y坐标映射（VR权重70% + 头部补偿30%）
        base_y = (corrected_vr * screen_h) * 0.7

        # 头部姿态补偿贡献30%
        if gaze_data.get("screen_coords_head"):
            head_x, head_y = gaze_data["screen_coords_head"]
            base_y += head_y * 0.3

        # 应用垂直偏移补偿
        base_y += calib["vertical_offset"] * calib["vertical_sens"]

        # 微小抖动模拟
        base_y += np.random.normal(0, 2)

        return (int(base_x + jitter), int(base_y + jitter))

    def _filter_coordinate_jumps(self, gaze_data):
        """过滤坐标跳变"""
        screen_w, screen_h = get_screen_resolution()
        if hasattr(self, 'prev_screen_coords') and gaze_data.get("screen_coords"):
            prev_coords = self.prev_screen_coords
            curr_coords = gaze_data["screen_coords"]
            # 计算两点间距离
            distance = np.sqrt((curr_coords[0] - prev_coords[0]) ** 2 + (curr_coords[1] - prev_coords[1]) ** 2)
            # 如果距离超过屏幕宽度的5%，则认为是跳变
            if distance > screen_w * 0.05:
                # 使用上一帧坐标平滑过渡
                gaze_data["screen_coords"] = (
                    int(prev_coords[0] * 0.7 + curr_coords[0] * 0.3),
                    int(prev_coords[1] * 0.7 + curr_coords[1] * 0.3)
                )

        # 更新上一帧坐标
        if gaze_data.get("screen_coords"):
            self.prev_screen_coords = gaze_data["screen_coords"]

    def _apply_filters(self, gaze_data):
        """应用卡尔曼滤波"""
        screen_w, screen_h = get_screen_resolution()
        if self.kf is not None and self.pupils_located:
            # 预测步骤
            self.kf.predict()

            # 准备观测向量
            head_pose_obs, gaze_obs = self._prepare_observations(gaze_data, screen_w, screen_h)

            # 在更新卡尔曼滤波器前，设置目标点和校准误差
            self._set_kalman_target_and_error(gaze_data)

            # 新增：动态调整卡尔曼参数
            self._adjust_kalman_parameters(gaze_data)

            # 只有当观测值有效时才更新卡尔曼滤波器
            if head_pose_obs is not None or gaze_obs is not None:
                self.kf.update(head_pose_obs, gaze_obs)

                # 获取滤波后的数据
                self._process_filtered_data(gaze_data, screen_w, screen_h)

    def _prepare_observations(self, gaze_data, screen_w, screen_h):
        """准备卡尔曼滤波观测值"""
        head_pose_obs = None
        gaze_obs = None

        # 头部姿态观测
        if gaze_data.get("head_pose"):
            head_pose = gaze_data["head_pose"]
            # 检查头部姿态数据是否有效（不包含NaN）
            if (head_pose is not None and
                    not any(np.isnan(x) if isinstance(x, (int, float)) else False for x in head_pose)):
                head_pose_obs = head_pose  # [pitch, yaw, roll]

        # 注视方向观测
        hr = gaze_data["hr"]
        vr = gaze_data["vr"]

        # 校验 gaze_obs 有效性，避免无效输入
        if (hr is not None and vr is not None and 0 <= hr <= 1 and 0 <= vr <= 1):
            h_angle, v_angle = self.gaze_direction_detector.get_gaze_angles(gaze_data)
            if not (np.isnan(h_angle) or np.isnan(v_angle)):
                screen_coords = gaze_data.get("screen_coords")
                if (screen_coords and all(coord is not None for coord in screen_coords) and
                        0 <= screen_coords[0] <= screen_w and 0 <= screen_coords[1] <= screen_h):
                    screen_x, screen_y = screen_coords
                    gaze_obs = [h_angle, v_angle, screen_x, screen_y]
                else:
                    # 无效坐标时，不传递screen_coords，避免滤波错误
                    if hasattr(self, 'prev_raw_screen_coords') and self.prev_raw_screen_coords:
                        gaze_obs = [h_angle, v_angle, self.prev_raw_screen_coords[0],
                                    self.prev_raw_screen_coords[1]]
                    else:
                        gaze_obs = [h_angle, v_angle, screen_w // 2, screen_h // 2]
            else:
                gaze_obs = None  # 角度无效时，不更新滤波
        else:
            gaze_obs = None

        return head_pose_obs, gaze_obs

    def _set_kalman_target_and_error(self, gaze_data):
        """设置卡尔曼滤波的目标点和误差"""
        if gaze_data.get("target_coords"):
            target_x, target_y = gaze_data["target_coords"]
            self.kf.gaze_filter.set_target_coords(target_x, target_y)  # 传递目标点到滤波器

            # 计算并传递校准误差
            if gaze_data.get("screen_coords"):
                screen_x, screen_y = gaze_data["screen_coords"]
                error = np.sqrt((screen_x - target_x) ** 2 + (screen_y - target_y) ** 2)
                self.kf.update_calibration_error(error)  # 传递误差到滤波器

    # 修改 _process_filtered_data 方法
    def _process_filtered_data(self, gaze_data, screen_w, screen_h):
        """处理滤波后的数据"""
        filtered_head_pose = self.kf.get_filtered_head_pose()

        # 获取滤波后的数据
        filtered_screen_coords = self.kf.get_filtered_screen_coords()

        # 使用新的专用屏幕坐标滤波器
        if gaze_data.get("screen_coords"):
            screen_x, screen_y = gaze_data["screen_coords"]
            if screen_x is not None and screen_y is not None:
                # 更新专用滤波器
                self.screen_coord_filter.update(screen_x, screen_y)
                # 获取滤波后的坐标
                filtered_screen_coords = self.screen_coord_filter.get_filtered_coords()

        if filtered_screen_coords and all(coord is not None for coord in filtered_screen_coords):
            # 多滤波融合 - 卡尔曼滤波后进行滑动平均
            self.filtered_coords_history.append(filtered_screen_coords)
            if len(self.filtered_coords_history) > self.sliding_window_size:
                self.filtered_coords_history.pop(0)

            # 应用滑动平均滤波
            smoothed_coords = self._apply_sliding_average_filter(self.filtered_coords_history)
            if smoothed_coords:
                filtered_screen_coords = smoothed_coords

            gaze_data["filtered_screen_x"] = int(filtered_screen_coords[0])
            gaze_data["filtered_screen_y"] = int(filtered_screen_coords[1])

            # 更新prev_filtered_coords
            if not hasattr(self, 'prev_filtered_coords'):
                self.prev_filtered_coords = (gaze_data["filtered_screen_x"], gaze_data["filtered_screen_y"])
            else:
                self.prev_filtered_coords = (gaze_data["filtered_screen_x"], gaze_data["filtered_screen_y"])
        else:
            # 滤波失效时，用原始坐标平滑
            if not hasattr(self, 'prev_filtered_coords'):
                self.prev_filtered_coords = (screen_w // 2, screen_h // 2)

            gaze_data["filtered_screen_x"] = int(
                gaze_data["screen_coords"][0] * 0.7 + self.prev_filtered_coords[0] * 0.3) if gaze_data.get(
                "screen_coords") else screen_w // 2
            gaze_data["filtered_screen_y"] = int(
                gaze_data["screen_coords"][1] * 0.7 + self.prev_filtered_coords[1] * 0.3) if gaze_data.get(
                "screen_coords") else screen_h // 2
            self.prev_filtered_coords = (gaze_data["filtered_screen_x"], gaze_data["filtered_screen_y"])

        # 将滤波后的数据添加到gaze_data中
        gaze_data["filtered_head_pose"] = filtered_head_pose

    def _process_3d_gaze_vector(self, gaze_data):
        """处理3D注视向量（增强版，包含生理特征信息）"""
        # 使用 GazeDirectionDetector 生成3D注视向量
        gaze_angles, gaze_vector = self.gaze_direction_detector.detect_gaze_direction(gaze_data)

        # 将3D注视向量写入 gaze_data
        gaze_data["gaze_vx"] = float(gaze_vector[0])
        gaze_data["gaze_vy"] = float(gaze_vector[1])
        gaze_data["gaze_vz"] = float(gaze_vector[2])

        # 添加动态权重信息到gaze_data（用于调试）
        if gaze_data.get("head_pose"):
            pitch, yaw, roll = gaze_data["head_pose"]
            theta_yaw = self.gaze_direction_detector.params.get('theta_yaw', 15.0)
            theta_pitch = self.gaze_direction_detector.params.get('theta_pitch', 15.0)
            alpha_t = self.gaze_direction_detector._dynamic_weight(yaw, theta_yaw)
            beta_t = self.gaze_direction_detector._dynamic_weight(pitch, theta_pitch)

            gaze_data["dynamic_alpha"] = alpha_t  # 水平方向动态权重
            gaze_data["dynamic_beta"] = beta_t  # 垂直方向动态权重

        # 添加生理特征影响因子信息
        physiological_features = gaze_data.get('physiological_features', {})
        if physiological_features:
            pupil_ellipticity = physiological_features.get('pupil_ellipticity_left', 1.0) \
                if gaze_data.get('left_pupil') else physiological_features.get('pupil_ellipticity_right', 1.0)
            eyelid_occlusion = physiological_features.get('eyelid_occlusion_left', 0.0) \
                if gaze_data.get('left_pupil') else physiological_features.get('eyelid_occlusion_right', 0.0)

            k1 = self.gaze_direction_detector.params.get('ellipticity_weight', 0.3)
            k2 = self.gaze_direction_detector.params.get('occlusion_weight', 0.2)
            phi = 1 + k1 * (1 - pupil_ellipticity) - k2 * eyelid_occlusion

            gaze_data["physiological_phi"] = phi  # 生理特征影响因子
            gaze_data["pupil_ellipticity"] = pupil_ellipticity
            gaze_data["eyelid_occlusion"] = eyelid_occlusion

    def _integrate_physiological_features(self, gaze_data):
        """整合生理特征"""
        physiological_features = {}

        # 1. 瞳孔椭圆度
        if self.eye_left and hasattr(self.eye_left, 'pupil_ellipticity'):
            physiological_features['pupil_ellipticity_left'] = self.eye_left.pupil_ellipticity
        if self.eye_right and hasattr(self.eye_right, 'pupil_ellipticity'):
            physiological_features['pupil_ellipticity_right'] = self.eye_right.pupil_ellipticity

        # 2. 眼睑遮挡率
        if self.eye_left and hasattr(self.eye_left, 'eyelid_occlusion_rate'):
            physiological_features['eyelid_occlusion_left'] = self.eye_left.eyelid_occlusion_rate
        if self.eye_right and hasattr(self.eye_right, 'eyelid_occlusion_rate'):
            physiological_features['eyelid_occlusion_right'] = self.eye_right.eyelid_occlusion_rate

        # 3. 主视眼优势度
        left_confidence = 0
        right_confidence = 0

        if self.eye_left and self.eye_left.pupil:
            left_confidence = self.eye_left.pupil.is_valid * getattr(self.eye_left.pupil, 'area_ratio', 0)

        if self.eye_right and self.eye_right.pupil:
            right_confidence = self.eye_right.pupil.is_valid * getattr(self.eye_right.pupil, 'area_ratio', 0)

        physiological_features['eye_dominance'] = left_confidence - right_confidence

        # 将生理特征添加到gaze_data
        gaze_data['physiological_features'] = physiological_features

    def _output_debug_info(self, gaze_data):
        """输出调试信息"""
        # 添加调试日志 - 输出整合后的生理特征
        physiological_features = gaze_data.get('physiological_features', {})
        if physiological_features:
            avg_ellipticity = np.mean([
                physiological_features.get('pupil_ellipticity_left', 1.0),
                physiological_features.get('pupil_ellipticity_right', 1.0)
            ])
            avg_occlusion = np.mean([
                physiological_features.get('eyelid_occlusion_left', 0.0),
                physiological_features.get('eyelid_occlusion_right', 0.0)
            ])
            print(f"[DEBUG] 整合生理特征 - 平均椭圆度: {avg_ellipticity:.3f}, 平均遮挡率: {avg_occlusion:.3f}")

        # 输出瞳孔核心数据
        self._output_pupil_debug_info(gaze_data)

        # 输出眼动与头部核心参数
        left_ratio = self.eye_left.blinking if self.eye_left and self.eye_left.blinking is not None else 0.0
        right_ratio = self.eye_right.blinking if self.eye_right and self.eye_right.blinking is not None else 0.0
        is_blinking = "是" if self.is_blinking() else "否"

        print(f"[GAZE DEBUG] 眨眼检测: {is_blinking} | 左眼比率:{left_ratio:.2f} | 右眼比率:{right_ratio:.2f}")

        if gaze_data.get("head_pose"):
            pitch, yaw, roll = gaze_data["head_pose"]
            print(f"[GAZE DEBUG] 头部姿态(Pitch,Yaw,Roll): ({pitch:.3f}, {yaw:.3f}, {roll:.3f})")

        if gaze_data["hr"] is not None and gaze_data["vr"] is not None:
            screen_coords = gaze_data.get("screen_coords", (0, 0))
            target_coords = gaze_data.get("target_coords", (0, 0))
            print(f"[DEBUG] 水平比率(HR):{gaze_data['hr']:.3f} | 垂直比率(VR):{gaze_data['vr']:.3f} | " +
                  f"屏幕坐标: ({screen_coords[0]}, {screen_coords[1]}) | 目标坐标: ({target_coords[0]}, {target_coords[1]})")

        # 添加动态权重调试信息
        if "dynamic_alpha" in gaze_data and "dynamic_beta" in gaze_data:
            print(f"[DEBUG] 动态权重 - 水平α(t):{gaze_data['dynamic_alpha']:.3f}, 垂直β(t):{gaze_data['dynamic_beta']:.3f}")

        # 添加生理特征影响因子调试信息
        if "physiological_phi" in gaze_data:
            print(f"[DEBUG] 生理特征影响因子 φ: {gaze_data['physiological_phi']:.3f}")
            print(f"[DEBUG] 瞳孔椭圆度: {gaze_data.get('pupil_ellipticity', 1.0):.3f}")
            print(f"[DEBUG] 眼睑遮挡率: {gaze_data.get('eyelid_occlusion', 0.0):.3f}")

        # 每5帧输出一次精简版Gaze数据
        if self.frame_count % 5 == 0:
            self._print_simplified_gaze_data(gaze_data)

    def _output_pupil_debug_info(self, gaze_data):
        """输出瞳孔调试信息"""
        left_coords = gaze_data.get('left_pupil')
        right_coords = gaze_data.get('right_pupil')

        # 输出瞳孔核心数据（双眼有效时输出）
        if left_coords and right_coords:
            left_radius = self.eye_left.pupil.radius if self.eye_left and self.eye_left.pupil else None
            right_radius = self.eye_right.pupil.radius if self.eye_right and self.eye_right.pupil else None
            # 从 pupil 对象获取 area_ratio，而不是 eye 对象，并处理 None 值
            left_area_ratio = self.eye_left.pupil.area_ratio if self.eye_left and self.eye_left.pupil and hasattr(
                self.eye_left.pupil, 'area_ratio') else None
            right_area_ratio = self.eye_right.pupil.area_ratio if self.eye_right and self.eye_right.pupil and hasattr(
                self.eye_right.pupil, 'area_ratio') else None

            # 修复格式化字符串中的条件表达式
            left_radius_str = f"{left_radius:.1f}" if left_radius is not None else "N/A"
            right_radius_str = f"{right_radius:.1f}" if right_radius is not None else "N/A"

            print(
                f"[PUPIL DEBUG] 左眼: 坐标({left_coords[0]:.1f}, {left_coords[1]:.1f}) | 半径{left_radius_str} | " +
                f"右眼: 坐标({right_coords[0]:.1f}, {right_coords[1]:.1f}) | 半径{right_radius_str}")

            # 安全地格式化 area_ratio 值
            left_area_str = f"{left_area_ratio:.3f}" if left_area_ratio is not None else "N/A"
            right_area_str = f"{right_area_ratio:.3f}" if right_area_ratio is not None else "N/A"
            print(f"[PUPIL DEBUG] 虹膜面积比 - 左眼:{left_area_str} | 右眼:{right_area_str}")

        # 输出单眼有效数据
        elif left_coords:
            left_radius = self.eye_left.pupil.radius if self.eye_left and self.eye_left.pupil else None
            # 从 pupil 对象获取 area_ratio，而不是 eye 对象，并处理 None 值
            left_area_ratio = self.eye_left.pupil.area_ratio if self.eye_left and self.eye_left.pupil and hasattr(
                self.eye_left.pupil, 'area_ratio') else None

            # 修复格式化字符串中的条件表达式
            left_radius_str = f"{left_radius:.1f}" if left_radius is not None else "N/A"

            print(
                f"[PUPIL DEBUG] 左眼: 坐标({left_coords[0]:.1f}, {left_coords[1]:.1f}) | 半径{left_radius_str}")

            # 安全地格式化 area_ratio 值
            left_area_str = f"{left_area_ratio:.3f}" if left_area_ratio is not None else "N/A"
            print(f"[PUPIL DEBUG] 虹膜面积比 - 左眼:{left_area_str}")

        elif right_coords:
            right_radius = self.eye_right.pupil.radius if self.eye_right and self.eye_right.pupil else None
            # 从 pupil 对象获取 area_ratio，而不是 eye 对象，并处理 None 值
            right_area_ratio = self.eye_right.pupil.area_ratio if self.eye_right and self.eye_right.pupil and hasattr(
                self.eye_right.pupil, 'area_ratio') else None

            # 修复格式化字符串中的条件表达式
            right_radius_str = f"{right_radius:.1f}" if right_radius is not None else "N/A"

            print(
                f"[PUPIL DEBUG] 右眼: 坐标({right_coords[0]:.1f}, {right_coords[1]:.1f}) | 半径{right_radius_str}")

            # 安全地格式化 area_ratio 值
            right_area_str = f"{right_area_ratio:.3f}" if right_area_ratio is not None else "N/A"
            print(f"[PUPIL DEBUG] 虹膜面积比 - 右眼:{right_area_str}")

    def _handle_dynamic_calibration(self, gaze_data):
        """处理动态校准"""
        # 检查是否需要进行动态校准
        if self._check_dynamic_calibration_needed(gaze_data):
            # 触发动态校准
            self._trigger_dynamic_calibration(gaze_data)

        # 如果正在进行动态校准，收集样本
        if self.dynamic_calibration_triggered:
            self._collect_dynamic_calibration_sample(gaze_data)

    def _check_dynamic_calibration_needed(self, gaze_data):
        """
        检查是否需要触发动态校准
        """
        screen_coords = gaze_data.get("screen_coords")
        target_coords = gaze_data.get("target_coords")

        # 只在追踪阶段检查（非校准阶段）
        if (hasattr(self, 'direction_calibrator') and
                screen_coords and target_coords and
                isinstance(screen_coords, (list, tuple)) and
                isinstance(target_coords, (list, tuple)) and
                len(screen_coords) >= 2 and len(target_coords) >= 2):

            # 计算误差距离
            error_distance = np.sqrt(
                (screen_coords[0] - target_coords[0]) ** 2 +
                (screen_coords[1] - target_coords[1]) ** 2
            )

            # 添加到误差历史
            self.error_history.append(error_distance)
            if len(self.error_history) > self.DYNAMIC_CALIBRATION_FRAMES:
                self.error_history.pop(0)

            # 检查是否连续多帧误差超过阈值
            if (len(self.error_history) >= self.DYNAMIC_CALIBRATION_FRAMES and
                    all(error > self.DYNAMIC_CALIBRATION_THRESHOLD for error in self.error_history)):
                return True

        return False

    def _trigger_dynamic_calibration(self, gaze_data):
        """
        触发动态校准流程
        """
        print("[INFO] 检测到持续误差，触发动态校准...")
        self.dynamic_calibration_triggered = True
        self.dynamic_calibration_samples = []

        # 设置动态校准点为屏幕中心
        screen_w, screen_h = get_screen_resolution()
        self.dynamic_calibration_point = (screen_w // 2, screen_h // 2)

        print(f"[INFO] 请注视屏幕中心点 ({self.dynamic_calibration_point[0]}, {self.dynamic_calibration_point[1]})")
        print(f"[INFO] 正在收集 {self.DYNAMIC_CALIBRATION_SAMPLES_NEEDED} 个样本...")

    def _collect_dynamic_calibration_sample(self, gaze_data):
        """
        收集动态校准样本
        """
        if (gaze_data.get("hr") is not None and
                gaze_data.get("vr") is not None and
                len(self.dynamic_calibration_samples) < self.DYNAMIC_CALIBRATION_SAMPLES_NEEDED):

            # 收集样本
            sample = {
                "target_point": (
                    self.dynamic_calibration_point[0],
                    self.dynamic_calibration_point[1]
                ),
                "hr_vr": (
                    gaze_data.get("hr"),
                    gaze_data.get("vr")
                ),
                "head_pose": (
                    gaze_data.get("head_pitch", 0),
                    gaze_data.get("head_yaw", 0),
                    gaze_data.get("head_roll", 0)
                ) if gaze_data.get("head_pose") else (0, 0, 0)
            }

            self.dynamic_calibration_samples.append(sample)
            print(
                f"[DEBUG] 收集动态校准样本 {len(self.dynamic_calibration_samples)}/{self.DYNAMIC_CALIBRATION_SAMPLES_NEEDED}")

            # 如果收集到足够样本，更新校准模型
            if len(self.dynamic_calibration_samples) >= self.DYNAMIC_CALIBRATION_SAMPLES_NEEDED:
                self._update_calibration_model()

    def _update_calibration_model(self):
        """
        更新校准模型
        """
        if len(self.dynamic_calibration_samples) >= 3:  # 至少需要3个样本
            try:
                # 计算平均 HR/VR 值
                hr_values = [sample["hr_vr"][0] for sample in self.dynamic_calibration_samples if
                             sample["hr_vr"][0] is not None]
                vr_values = [sample["hr_vr"][1] for sample in self.dynamic_calibration_samples if
                             sample["hr_vr"][1] is not None]

                if hr_values and vr_values:
                    avg_hr = np.mean(hr_values)
                    avg_vr = np.mean(vr_values)

                    # 归一化目标点坐标
                    screen_w, screen_h = get_screen_resolution()
                    normalized_target = (
                        self.dynamic_calibration_point[0] / screen_w,
                        self.dynamic_calibration_point[1] / screen_h
                    )

                    # 调用方向校准器的动态更新方法
                    if hasattr(self.direction_calibrator, 'update_calibration_dynamically'):
                        self.direction_calibrator.update_calibration_dynamically(
                            normalized_target,
                            (avg_hr, avg_vr)
                        )
                        print("[INFO] 动态校准模型更新完成")

            except Exception as e:
                print(f"[ERROR] 更新校准模型时出错: {e}")
        else:
            print("[WARN] 动态校准样本不足，无法更新模型")

        # 重置动态校准状态
        self.dynamic_calibration_triggered = False
        self.dynamic_calibration_samples = []
        self.error_history = []  # 清空误差历史
        self.dynamic_calibration_point = None


    def _print_simplified_gaze_data(self, gaze_data):
        """打印精简版的gaze_data"""
        print("----------------------------------------")
        print(f"[精简Gaze数据] 帧号:{self.frame_count} | 时间戳: {gaze_data['timestamp']}")

        # 1. 瞳孔信息
        left_pupil = gaze_data.get('left_pupil')
        right_pupil = gaze_data.get('right_pupil')
        left_radius = gaze_data.get('left_pupil_radius')
        right_radius = gaze_data.get('right_pupil_radius')

        left_info = f"坐标({left_pupil[0]:.3f}, {left_pupil[1]:.3f}) | 半径{left_radius:.3f}" if left_pupil and left_radius else "无数据"
        right_info = f"坐标({right_pupil[0]:.3f}, {right_pupil[1]:.3f}) | 半径{right_radius:.3f}" if right_pupil and right_radius else "无数据"
        print(f"1. 瞳孔信息")
        print(f"   左眼: {left_info} | 右眼: {right_info}")

        # 2. 眼动参数
        hr = gaze_data.get('hr')
        vr = gaze_data.get('vr')
        is_blinking = "是" if gaze_data.get('is_blinking') else "否"
        screen_coords = gaze_data.get('screen_coords', (0, 0))

        # 安全地格式化HR和VR值
        hr_str = f"{hr:.3f}" if hr is not None else "N/A"
        vr_str = f"{vr:.3f}" if vr is not None else "N/A"
        screen_x_str = f"{screen_coords[0]}" if screen_coords and screen_coords[0] is not None else "N/A"
        screen_y_str = f"{screen_coords[1]}" if screen_coords and screen_coords[1] is not None else "N/A"

        print(f"2. 眼动参数")
        print(
            f"   水平比率(HR):{hr_str} | 垂直比率(VR):{vr_str} | 是否眨眼: {is_blinking} | 屏幕坐标: ({screen_x_str}, {screen_y_str})")

        # 3. 头部姿态
        head_pose = gaze_data.get('head_pose')
        filtered_head_pose = gaze_data.get('filtered_head_pose')
        if head_pose:
            pitch, yaw, roll = head_pose
            print(f"3. 头部姿态")
            print(f"   Pitch:{pitch:.3f} | Yaw:{yaw:.3f} | Roll:{roll:.3f} | 滤波后姿态: {filtered_head_pose}")
        else:
            print(f"3. 头部姿态: 无有效数据")

        # 4. 关键特征点
        left_inner = (gaze_data.get('left_eye_inner_x'), gaze_data.get('left_eye_inner_y'))
        left_outer = (gaze_data.get('left_eye_outer_x'), gaze_data.get('left_eye_outer_y'))
        right_inner = (gaze_data.get('right_eye_inner_x'), gaze_data.get('right_eye_inner_y'))
        right_outer = (gaze_data.get('right_eye_outer_x'), gaze_data.get('right_eye_outer_y'))

        if all(v is not None for v in left_inner + left_outer + right_inner + right_outer):
            print(f"4. 关键特征点")
            print(f"   左眼内外角: ({left_inner[0]},{left_inner[1]})/({left_outer[0]},{left_outer[1]}) | " +
                  f"右眼内外角: ({right_inner[0]},{right_inner[1]})/({right_outer[0]},{right_outer[1]})")
        else:
            print(f"4. 关键特征点: 无有效数据")

        # 5. 滤波后信息
        filtered_x = gaze_data.get('filtered_screen_x')
        filtered_y = gaze_data.get('filtered_screen_y')
        filtered_x_str = f"{filtered_x}" if filtered_x is not None else "N/A"
        filtered_y_str = f"{filtered_y}" if filtered_y is not None else "N/A"
        print(f"5. 滤波后信息")
        print(f"   filtered_head_pose  : {filtered_head_pose} | " +
              f"filtered_screen_x   : {filtered_x_str} | " +
              f"filtered_screen_y   : {filtered_y_str}")
        print("----------------------------------------")

    # 在 GazeTracking 类中添加一个辅助方法来设置默认的眼部特征值
    def _set_default_eye_features(self, gaze_data):
        """设置默认的眼部特征值"""
        default_keys = {
            "left_eye_inner_x": 0, "left_eye_inner_y": 0,
            "left_eye_outer_x": 0, "left_eye_outer_y": 0,
            "right_eye_inner_x": 0, "right_eye_inner_y": 0,
            "right_eye_outer_x": 0, "right_eye_outer_y": 0,
            "left_eye_upper_lid_x": 0, "left_eye_upper_lid_y": 0,
            "left_eye_lower_lid_x": 0, "left_eye_lower_lid_y": 0,
            "right_eye_upper_lid_x": 0, "right_eye_upper_lid_y": 0,
            "right_eye_lower_lid_x": 0, "right_eye_lower_lid_y": 0,
        }

        for key, default_val in default_keys.items():
            if gaze_data.get(key) is None:
                gaze_data[key] = default_val

    # 在 core/gaze_tracking.py 文件的 GazeTracking 类中添加新的辅助方法

    def _apply_sliding_average_filter(self, coords_list):
        """
        应用滑动平均滤波器

        参数:
            coords_list: 坐标历史列表 [(x1, y1), (x2, y2), ...]

        返回:
            smoothed_coords: 平滑后的坐标 (x, y)
        """
        if not coords_list:
            return None

        window_size = len(coords_list)

        if window_size == 1:
            return coords_list[0]
        elif window_size == 2:
            # 两帧加权平均：最新帧权重0.6，前一帧权重0.4
            x = int(coords_list[-1][0] * 0.6 + coords_list[-2][0] * 0.4)
            y = int(coords_list[-1][1] * 0.6 + coords_list[-2][1] * 0.4)
            return (x, y)
        else:  # window_size >= 3
            # 三帧加权平均：最新帧权重0.5，前两帧各0.25
            x = int(coords_list[-1][0] * 0.5 +
                    coords_list[-2][0] * 0.25 +
                    coords_list[-3][0] * 0.25)
            y = int(coords_list[-1][1] * 0.5 +
                    coords_list[-2][1] * 0.25 +
                    coords_list[-3][1] * 0.25)
            return (x, y)

        # 在 core/gaze_tracking.py 的 GazeTracking 类中添加以下方法

        def check_calibration_trigger(self, current_rmse, camera_id):
            """
            检查是否需要触发校准

            参数:
                current_rmse: 当前误差值（像素）
                camera_id: 当前摄像头ID

            返回:
                bool: 是否需要触发校准
            """
            # 1. 误差超标触发（对应创新点1"误差稳定性"标准）
            if current_rmse > 50:
                print("校准触发：当前误差{}px>50px，请重新校准".format(current_rmse))
                return True

            # 2. 设备变动触发（检测摄像头ID变化，避免跨设备偏差）
            last_camera_id = self.load_last_camera_id()  # 读取历史摄像头ID
            if camera_id != last_camera_id:
                print("校准触发：检测到新设备，需重新校准")
                self.save_last_camera_id(camera_id)  # 更新历史ID
                return True

            return False

        def load_last_camera_id(self):
            """加载历史摄像头ID"""
            calib_path = os.path.join(os.path.dirname(__file__), "calibration/camera_id.txt")
            if os.path.exists(calib_path):
                try:
                    with open(calib_path, 'r') as f:
                        return f.read().strip()
                except Exception as e:
                    print(f"[CALIB ERROR] 加载摄像头ID失败: {e}")
            return None

        def save_last_camera_id(self, camera_id):
            """保存当前摄像头ID"""
            calib_path = os.path.join(os.path.dirname(__file__), "calibration/camera_id.txt")
            try:
                os.makedirs(os.path.dirname(calib_path), exist_ok=True)
                with open(calib_path, 'w') as f:
                    f.write(str(camera_id))
            except Exception as e:
                print(f"[CALIB ERROR] 保存摄像头ID失败: {e}")


    def _adjust_kalman_parameters(self, gaze_data):
        """动态调整卡尔曼滤波参数"""
        try:
            # 获取当前头部姿态
            current_head_pose = gaze_data.get("head_pose")
            if current_head_pose is None:
                return

            current_head = {
                "yaw": current_head_pose[1],
                "pitch": current_head_pose[0],
                "roll": current_head_pose[2]
            }

            # 获取上一帧头部姿态
            if hasattr(self, '_last_head_pose'):
                last_head = {
                    "yaw": self._last_head_pose[1],
                    "pitch": self._last_head_pose[0],
                    "roll": self._last_head_pose[2]
                }
            else:
                # 如果没有历史数据，使用当前姿态作为初始值
                last_head = current_head.copy()

            # 计算误差
            rmse = 0
            if gaze_data.get("screen_coords") and gaze_data.get("target_coords"):
                screen_x, screen_y = gaze_data["screen_coords"]
                target_x, target_y = gaze_data["target_coords"]
                rmse = np.sqrt((screen_x - target_x) ** 2 + (screen_y - target_y) ** 2)

            # 调整卡尔曼参数
            if hasattr(self.kf, 'adjust_kalman_params'):
                self.kf.adjust_kalman_params(current_head, last_head, rmse)

            # 保存当前头部姿态作为下一帧的历史数据
            self._last_head_pose = current_head_pose

        except Exception as e:
            print(f"[KALMAN] 参数调整失败: {e}")
