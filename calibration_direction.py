# direction_calibrator.py
from __future__ import division
import numpy as np

from util.utils import get_screen_resolution
import cv2

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor


class EnhancedGazeMapper:
    # 在 EnhancedGazeMapper.__init__ 方法中替换原有模型初始化代码
    def __init__(self, use_nn=False, primary_eye='right'):
        """
        初始化注视点映射器

        参数:
            use_nn: 是否使用神经网络 (默认False，使用梯度提升树)
            primary_eye: 主视眼 ('left' 或 'right')，默认为 'right'
        """
        self.use_nn = use_nn
        self.primary_eye = primary_eye

        # 特征名称
        self.feature_names = [
            'norm_dx', 'norm_dy', 'norm_radius',
            'head_pitch', 'head_yaw', 'head_roll',
            'pupil_radius_change', 'eye_openness'  # 新增特征
        ]

        if use_nn:
            # 使用梯度提升树
            from sklearn.ensemble import GradientBoostingRegressor
            self.model_x = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42
            )
            self.model_y = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42
            )
        else:
            # 使用多项式回归 + 岭回归
            self.pipeline_x = Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('ridge', Ridge(alpha=1.0))
            ])
            self.pipeline_y = Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('ridge', Ridge(alpha=1.0))
            ])

    def set_primary_eye(self, eye_side):
        """
        设置主视眼

        参数:
            eye_side: 'left' 或 'right'
        """
        if eye_side in ['left', 'right']:
            self.primary_eye = eye_side
        else:
            raise ValueError("eye_side must be 'left' or 'right'")

    # 在 EnhancedGazeMapper.extract_features 方法中增加新特征
    def extract_features(self, gaze_data):
        """
        从gaze_data中提取特征

        参数:
            gaze_data: 包含眼部和头部数据的字典

        返回:
            features_array: 特征数组
        """
        features = []

        # 根据主视眼选择数据
        if self.primary_eye == 'left':
            norm_dx = gaze_data.get('left_norm_dx', 0)
            norm_dy = gaze_data.get('left_norm_dy', 0)
            norm_radius = gaze_data.get('left_norm_radius', 0)
            pupil_radius = gaze_data.get('left_pupil_radius', 0)
        else:  # 默认右眼
            norm_dx = gaze_data.get('right_norm_dx', 0)
            norm_dy = gaze_data.get('right_norm_dy', 0)
            norm_radius = gaze_data.get('right_norm_radius', 0)
            pupil_radius = gaze_data.get('right_pupil_radius', 0)

        # 头部姿态数据
        head_pitch = gaze_data.get('head_pitch', 0)
        head_yaw = gaze_data.get('head_yaw', 0)
        head_roll = gaze_data.get('head_roll', 0)

        # 新增特征：瞳孔大小变化率（需要历史数据支持）
        pupil_radius_change = gaze_data.get('pupil_radius_change', 0)  # 当前帧与前几帧的半径差

        # 新增特征：眼睑开合度（上下眼睑距离）
        if self.primary_eye == 'left':
            upper_lid_y = gaze_data.get('left_eye_upper_lid_y', 0)
            lower_lid_y = gaze_data.get('left_eye_lower_lid_y', 0)
        else:
            upper_lid_y = gaze_data.get('right_eye_upper_lid_y', 0)
            lower_lid_y = gaze_data.get('right_eye_lower_lid_y', 0)

        eye_openness = abs(upper_lid_y - lower_lid_y) if upper_lid_y is not None and lower_lid_y is not None else 0

        # 构建特征向量
        feature_vector = [
            norm_dx, norm_dy, norm_radius,
            head_pitch, head_yaw, head_roll,
            pupil_radius_change, eye_openness
        ]

        return np.array(feature_vector).reshape(1, -1)

    def fit(self, X, y):
        """
        训练模型

        参数:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标坐标 (n_samples, 2) - [x, y] 屏幕坐标
        """
        if self.use_nn:
            self.model_x.fit(X, y[:, 0])
            self.model_y.fit(X, y[:, 1])
        else:
            self.pipeline_x.fit(X, y[:, 0])
            self.pipeline_y.fit(X, y[:, 1])

    def predict(self, X):
        """
        预测注视点

        参数:
            X: 特征矩阵 (n_samples, n_features)

        返回:
            predicted_coords: 预测的屏幕坐标 (n_samples, 2)
        """
        if self.use_nn:
            x_pred = self.model_x.predict(X)
            y_pred = self.model_y.predict(X)
        else:
            x_pred = self.pipeline_x.predict(X)
            y_pred = self.pipeline_y.predict(X)
        return np.column_stack((x_pred, y_pred))

    def fit_from_gaze_data(self, gaze_data_list, screen_coords_list):
        """
        从gaze_data列表训练模型

        参数:
            gaze_data_list: gaze_data字典列表
            screen_coords_list: 对应的屏幕坐标列表
        """
        # 新增：计算用户个性化阈值
        all_yaw = [g["head_yaw"] for g in gaze_data_list]
        all_pitch = [g["head_pitch"] for g in gaze_data_list]
        self.theta_yaw = (max(all_yaw) - min(all_yaw)) / 2  # 个性化yaw阈值
        self.theta_pitch = (max(all_pitch) - min(all_pitch)) / 2  # 个性化pitch阈值

        # 提取特征
        X_features = []
        for gaze_data in gaze_data_list:
            features = self.extract_features(gaze_data)
            X_features.append(features.flatten())

        X = np.array(X_features)
        y = np.array(screen_coords_list)

        # 训练模型
        self.fit(X, y)

    # 动态权重计算
    def _get_dynamic_weights(self, gaze_data):
        yaw = gaze_data["head_yaw"]
        pitch = gaze_data["head_pitch"]
        alpha = min(0.5, max(0.1, abs(yaw) / self.theta_yaw))  # 头部权重
        beta = 1 - alpha  # 眼部权重
        return alpha, beta

    def predict_from_gaze_data(self, gaze_data):
        """
        从单个gaze_data预测注视点

        参数:
            gaze_data: 单个gaze_data字典

        返回:
            predicted_coords: 预测的屏幕坐标 [x, y]
        """
        features = self.extract_features(gaze_data)
        return self.predict(features)[0]


class Calibration:
    """
    校准类，用于寻找最佳阈值以提高瞳孔检测精度
    """

    def __init__(self):
        self.nb_frames = 20  # 需要采集的帧数
        self.thresholds_left = []
        self.thresholds_right = []

    def is_complete(self):
        return len(self.thresholds_left) >= self.nb_frames and len(self.thresholds_right) >= self.nb_frames

    def threshold(self, side):
        """返回指定眼的平均阈值"""
        if side == 0:
            if not self.thresholds_left:
                return 40  # 默认阈值
            return int(sum(self.thresholds_left) / len(self.thresholds_left))
        elif side == 1:
            if not self.thresholds_right:
                return 40
            return int(sum(self.thresholds_right) / len(self.thresholds_right))

    @staticmethod
    def iris_size(frame):
        """计算瞳孔区域占眼睛区域的比例"""
        frame = frame[5:-5, 5:-5]  # 去除边缘干扰
        height, width = frame.shape[:2]
        nb_pixels = height * width
        nb_blacks = nb_pixels - cv2.countNonZero(frame)
        return nb_blacks / nb_pixels

    @staticmethod
    def find_best_threshold(eye_frame):
        """找出最优二值化阈值"""
        average_iris_size = 0.48  # 理想瞳孔面积占比
        trials = {}

        for threshold in range(5, 100, 5):
            # 使用 Pupil.image_processing 模拟处理
            processed = Pupil.image_processing(eye_frame, threshold)
            trials[threshold] = Calibration.iris_size(processed)

        best_threshold, iris_size = min(trials.items(), key=lambda p: abs(p[1] - average_iris_size))
        print(f"Best threshold: {best_threshold}, Iris Area Ratio: {iris_size:.4f}\n")
        return best_threshold

    def evaluate(self, eye_frame, side):
        """添加样本到对应的眼睛"""
        threshold = self.find_best_threshold(eye_frame)
        if side == 0:
            self.thresholds_left.append(threshold)
        elif side == 1:
            self.thresholds_right.append(threshold)


class Pupil:
    """
    瞳孔图像预处理和分割模型模拟类（仅保留 image_processing 方法供 Calibration 调用）
    """

    @staticmethod
    def image_processing(eye_frame, threshold):
        """简单的图像处理方法模拟"""
        kernel = np.ones((3, 3), np.uint8)
        new_frame = cv2.bilateralFilter(eye_frame, d=15, sigmaColor=100, sigmaSpace=100)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        new_frame = clahe.apply(new_frame)
        new_frame = cv2.erode(new_frame, kernel, iterations=2)
        new_frame = cv2.dilate(new_frame, kernel, iterations=1)
        new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)[1]
        return new_frame


class DirectionCalibrator:
    """
    注视方向校准器，基于目标点与瞳孔位置/视线比例进行方向映射
    """

    def __init__(self, primary_eye='right'):
        """
        初始化方向校准器

        参数:
            primary_eye: 主视眼 ('left' 或 'right')，默认为 'right'
        """
        # 瞳孔定位相关（保留）
        self.primary_eye = primary_eye
        self.center_pupil = None  # 瞳孔中心基准点
        self.samples_pupil = []  # 瞳孔样本
        self.calibrated_pupil = False  # 瞳孔校准是否完成

        # 添加调试信息存储
        self.calibration_debug = []  # 新增：存储完整的调试数据

        # 回归模型（可选保留）
        self.regression_model = None  # 用于屏幕坐标预测的模型

        # 校准核心字段
        self.calibration_samples = []  # 统一保存所有校准样本
        self.calibration_ratios = []  # 保存 (target_point, eye_ratio) 对，用于存储比例校准样本
        self.calibrated_ratio = False  # 确保初始状态为 False
        self.current_target = None  # 当前校准的目标点
        self.calibrated_ratio = False  # 是否已完成校准

        # 添加回归模型
        self.regression_model_x = None
        self.regression_model_y = None
        self.poly_features = None

        # 初始化 gaze_mapper
        self.gaze_mapper = EnhancedGazeMapper(primary_eye=primary_eye)

        self.debug_data = []

        # 新增：存储包含头部姿态的校准数据
        self.calibration_with_head_pose = []

        # 新增：头部姿态补偿模型
        self.head_pose_model = None

        self.user_physiological_features = {}  # 存储用户生理特征

        # 新增：头部姿态区域划分
        self.head_pose_zones = {
            'center': {'yaw_range': (-5, 5), 'pitch_range': (-5, 5)},
            'left': {'yaw_range': (-30, -5), 'pitch_range': (-5, 5)},
            'right': {'yaw_range': (5, 30), 'pitch_range': (-5, 5)},
            'up': {'yaw_range': (-5, 5), 'pitch_range': (5, 30)},
            'down': {'yaw_range': (-5, 5), 'pitch_range': (-30, -5)}
        }

        # 新增：存储按头部姿态划分的样本
        self.head_pose_zone_samples = {zone: [] for zone in self.head_pose_zones.keys()}

        # 新增：头部姿态区域专属模型
        self.head_pose_zone_models = {zone: None for zone in self.head_pose_zones.keys()}

    def set_primary_eye(self, eye_side):
        """
        设置主视眼

        参数:
            eye_side: 'left' 或 'right'
        """
        if eye_side in ['left', 'right']:
            self.primary_eye = eye_side
            self.gaze_mapper.set_primary_eye(eye_side)
        else:
            raise ValueError("eye_side must be 'left' or 'right'")

    def add_sample_with_debug(self, hr, vr, debug_entry):
        # 添加调试输出
        print(f"[DEBUG] 添加样本 - HR: {hr}, VR: {vr}")
        print(f"[DEBUG] 样本内容: {debug_entry}")

        # 确保添加到列表
        self.calibration_samples.append(debug_entry)

        # 添加计数验证
        print(f"[DEBUG] 当前样本数: {len(self.calibration_samples)}")

    # 封装调试数据构建逻辑
    def create_debug_entry(self, gaze_data, log_data, target_x, target_y, screen_w, screen_h, screen_coords):
        return {
            "target_point": (target_x / screen_w, target_y / screen_h),
            "eye_ratio": (gaze_data["hr"], gaze_data["vr"]),
            "timestamp": log_data["timestamp"],
            "head_pose": (gaze_data["head_pitch"], gaze_data["head_yaw"], gaze_data["head_roll"]),
            "gaze_vector": (gaze_data["gaze_vx"], gaze_data["gaze_vy"], gaze_data["gaze_vz"]),
            "pupil_coords": {
                "left": (gaze_data["left_pupil_x"], gaze_data["left_pupil_y"]),
                "right": (gaze_data["right_pupil_x"], gaze_data["right_pupil_y"]),
            },
            "eye_features": {
                "left_eye_inner": (gaze_data["left_eye_inner_x"], gaze_data["left_eye_inner_y"]),
                "left_eye_outer": (gaze_data["left_eye_outer_x"], gaze_data["left_eye_outer_y"]),
                "right_eye_inner": (gaze_data["right_eye_inner_x"], gaze_data["right_eye_inner_y"]),
                "right_eye_outer": (gaze_data["right_eye_outer_x"], gaze_data["right_eye_outer_y"]),
                "left_eye_upper_lid": (gaze_data["left_eye_upper_lid_x"], gaze_data["left_eye_upper_lid_y"]),
                "left_eye_lower_lid": (gaze_data["left_eye_lower_lid_x"], gaze_data["left_eye_lower_lid_y"]),
                "right_eye_upper_lid": (gaze_data["right_eye_upper_lid_x"], gaze_data["right_eye_upper_lid_y"]),
                "right_eye_lower_lid": (gaze_data["right_eye_lower_lid_x"], gaze_data["right_eye_lower_lid_y"]),
            },
            "is_blinking": gaze_data["is_blinking"],
            "screen_coords": screen_coords,
            "target_coords": (target_x, target_y),
        }

    def get_direction(self, left_pupil, right_pupil, threshold=15):
        """根据瞳孔偏移判断方向"""
        if not self.calibrated_pupil:
            return "Unknown"

        curr_x = (left_pupil[0] + right_pupil[0]) / 2 if left_pupil and right_pupil else None
        curr_y = (left_pupil[1] + right_pupil[1]) / 2 if left_pupil and right_pupil else None

        if curr_x is None or curr_y is None:
            return "Unknown"

        dx = curr_x - self.center_pupil[0]
        dy = curr_y - self.center_pupil[1]

        directions = []
        if dy > threshold:
            directions.append("Down")
        elif dy < -threshold:
            directions.append("Up")

        if dx > threshold:
            directions.append("Right")
        elif dx < -threshold:
            directions.append("Left")

        # 调试信息
        print(f"Calibrated Pupil: {self.calibrated_pupil}")
        print(f"Left Pupil: {left_pupil}, Right Pupil: {right_pupil}")
        print(f"Center Pupil: {self.center_pupil}")

        return "Looking Center" if not directions else "Looking " + " & ".join(directions)

    def add_sample_by_ratio(self, x_ratio, y_ratio):
        """添加基于比例的样本"""
        if self.current_target is not None:
            self.calibration_ratios.append([self.current_target, (x_ratio, y_ratio)])

    def get_direction_by_ratio(self, hr, vr, threshold=0.02):
        """根据视线比例判断最接近的校准点"""
        if not self.calibrated_ratio or hr is None or vr is None:
            return "Unknown"

        min_diff = float('inf')
        best_index = -1

        for i, (target_point, eye_ratio) in enumerate(self.calibration_ratios):
            ref_hr, ref_vr = eye_ratio
            diff = abs(ref_hr - hr) + abs(ref_vr - vr)
            if diff < min_diff and diff < threshold:
                min_diff = diff
                best_index = i

        return f"Point {best_index + 1}" if best_index >= 0 else "Unknown"

    def add_calibration_sample(self, target_point, eye_ratio, head_pose=None, physiological_features=None):
        """
        添加校准样本，支持头部姿态信息和生理特征
        """
        if target_point and eye_ratio:
            hr, vr = eye_ratio
            # 放宽样本有效性判断：HR/VR 非 None 即可
            if hr is not None and vr is not None and 0.01 <= hr <= 0.99 and 0.01 <= vr <= 0.99:
                # 检查与已有样本的距离，避免重复添加过于接近的样本
                if len(self.calibration_ratios) > 0:
                    # 处理混合数据格式的重复检查
                    existing_ratios = []
                    for sample in self.calibration_ratios:
                        if isinstance(sample, dict):
                            # 新格式：字典形式
                            existing_ratios.append(sample['eye_ratio'])
                        elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
                            # 旧格式：列表形式
                            existing_ratios.append(sample[1])  # eye_ratio 是第二个元素

                    if existing_ratios:
                        existing_ratios = np.array(existing_ratios)
                        distances = np.sqrt(np.sum((existing_ratios - np.array([hr, vr])) ** 2, axis=1))
                        min_distance = np.min(distances) if len(distances) > 0 else float('inf')
                        # 如果与最近样本距离太近，则不添加
                        if min_distance < 0.02:  # 调整阈值
                            print(f"[DEBUG] 样本过于接近已有样本，跳过添加")
                            return

                # 样本中加入头部姿态和生理特征
                sample = {
                    'target_point': target_point,
                    'eye_ratio': eye_ratio,
                    'head_pose': head_pose,
                    'physiological_features': physiological_features or {}
                }
                self.calibration_ratios.append(sample)
                self.calibration_with_head_pose.append(sample)  # 用于模型训练
                self.calibrated_ratio = len(self.calibration_ratios) >= 9  # 至少9个样本即视为校准完成
                print(f"[DEBUG] 有效样本添加: target={target_point}, ratio={eye_ratio}, head_pose={head_pose}")
            else:
                print(f"[WARN] 跳过无效样本: hr={hr}, vr={vr}")

    def finalize_ratio_calibration(self):
        """完成比例校准并训练回归模型（改进版，支持头部姿态特征和分区校准）"""
        try:
            # 1. 提取样本：包含头部姿态的特征
            targets = []
            features = []
            head_poses = []  # 新增：存储头部姿态用于分区

            for sample in self.calibration_ratios:
                # 统一处理不同格式的样本
                if isinstance(sample, dict):
                    # 新格式：字典形式
                    target_point = sample.get('target_point')
                    eye_ratio = sample.get('eye_ratio')
                    head_pose = sample.get('head_pose')
                    physiological_features = sample.get('physiological_features', {})
                elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
                    # 旧格式：列表形式
                    target_point = sample[0]
                    eye_ratio = sample[1]
                    head_pose = sample[2] if len(sample) >= 3 else None
                    physiological_features = {}
                else:
                    continue  # 跳过无效样本

                if target_point is None or eye_ratio is None:
                    continue

                hr, vr = eye_ratio
                # 放宽样本有效性判断：HR/VR 非 None 即可
                if hr is not None and vr is not None and 0.01 <= hr <= 0.99 and 0.01 <= vr <= 0.99:
                    targets.append(target_point)

                    # 提取生理特征
                    pupil_ellipticity = physiological_features.get('pupil_ellipticity_left',
                                                                   1.0) if self.primary_eye == 'left' else physiological_features.get(
                        'pupil_ellipticity_right', 1.0)
                    eyelid_occlusion = physiological_features.get('eyelid_occlusion_left',
                                                                  0.0) if self.primary_eye == 'left' else physiological_features.get(
                        'eyelid_occlusion_right', 0.0)
                    eye_dominance = physiological_features.get('eye_dominance', 0.0)

                    # 特征：HR + VR + 头部姿态（pitch/yaw/roll）+ 生理特征，共8个特征
                    feature = [hr, vr,
                               head_pose[0] if head_pose else 0,
                               head_pose[1] if head_pose else 0,
                               head_pose[2] if head_pose else 0,
                               pupil_ellipticity,
                               eyelid_occlusion,
                               eye_dominance]
                    features.append(feature)
                    head_poses.append(head_pose)  # 存储头部姿态用于分区

                    # 添加调试日志 - 模型输入维度验证
                    print(f"[DEBUG] 模型输入维度: {len(feature)} | HR={hr:.3f}, VR={vr:.3f}, "
                          f"椭圆度={pupil_ellipticity:.3f}, 遮挡率={eyelid_occlusion:.3f}, 优势度={eye_dominance:.3f}")

            if len(features) < 9:
                print("[WARN] 有效样本不足，需至少9个样本")
                self.calibrated_ratio = False
                return

            targets = np.array(targets)
            features = np.array(features)
            head_poses = np.array(head_poses)

            # 2. 数据清洗：剔除头部姿态异常的样本（如yaw>30）
            valid_idx = np.where(np.abs(features[:, 3]) <= 30)  # yaw（头部左右转）不超过30度
            features = features[valid_idx]
            targets = targets[valid_idx]
            head_poses = head_poses[valid_idx]

            if len(features) < 9:
                print("[WARN] 有效样本不足，需重新校准")
                self.calibrated_ratio = False
                return

            # 3. 按头部姿态区间分组
            # 定义头部姿态区间（可以根据实际数据调整）
            self.head_pose_zones = {
                'center': {'yaw_range': (-10, 10), 'pitch_range': (-10, 10)},
                'left': {'yaw_range': (-30, -10), 'pitch_range': (-15, 15)},
                'right': {'yaw_range': (10, 30), 'pitch_range': (-15, 15)},
                'up': {'yaw_range': (-15, 15), 'pitch_range': (10, 30)},
                'down': {'yaw_range': (-15, 15), 'pitch_range': (-30, -10)}
            }

            # 为每个区域训练独立模型
            self.zone_models = {}
            for zone_name, zone_range in self.head_pose_zones.items():
                yaw_min, yaw_max = zone_range['yaw_range']
                pitch_min, pitch_max = zone_range['pitch_range']

                # 筛选该区域的样本
                zone_indices = np.where(
                    (head_poses[:, 1] >= yaw_min) & (head_poses[:, 1] <= yaw_max) &
                    (head_poses[:, 0] >= pitch_min) & (head_poses[:, 0] <= pitch_max)
                )[0]

                if len(zone_indices) >= 3:  # 至少需要3个样本
                    zone_features = features[zone_indices]
                    zone_targets = targets[zone_indices]

                    # 训练该区域的模型
                    from sklearn.ensemble import RandomForestRegressor
                    zone_model_x = RandomForestRegressor(n_estimators=30, random_state=42, max_depth=8)
                    zone_model_y = RandomForestRegressor(n_estimators=30, random_state=42, max_depth=8)

                    zone_model_x.fit(zone_features, zone_targets[:, 0])
                    zone_model_y.fit(zone_features, zone_targets[:, 1])

                    self.zone_models[zone_name] = {
                        'model_x': zone_model_x,
                        'model_y': zone_model_y,
                        'sample_count': len(zone_indices)
                    }
                    print(f"[INFO] 区域 '{zone_name}' 模型训练完成，使用 {len(zone_indices)} 个样本")
                else:
                    print(f"[WARN] 区域 '{zone_name}' 样本不足 ({len(zone_indices)} 个)，跳过训练")

            # 4. 训练全局模型作为备选
            from sklearn.ensemble import RandomForestRegressor
            self.regression_model_x = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
            self.regression_model_y = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
            self.regression_model_x.fit(features, targets[:, 0])  # 预测X坐标
            self.regression_model_y.fit(features, targets[:, 1])  # 预测Y坐标

            self.calibrated_ratio = True
            print(f"[INFO] 头部姿态感知模型训练完成，使用 {len(features)} 个样本")

        except Exception as e:
            print(f"[ERROR] 校准模型训练失败: {e}")
            import traceback
            traceback.print_exc()
            self.calibrated_ratio = False

    def _finalize_ratio_calibration_original(self):
        """原始的校准逻辑（作为回退方案）"""
        try:
            # 1. 提取并清洗数据：剔除VR异常样本（HR和VR不在0.05~0.95范围内）
            valid_samples = []
            for sample in self.calibration_ratios:
                if len(sample) >= 2:  # 确保样本格式正确
                    target = sample[0]
                    hr_vr = sample[1]
                    if isinstance(hr_vr, (list, tuple)) and len(hr_vr) >= 2:
                        hr, vr = hr_vr
                        if 0.05 <= hr <= 0.95 and 0.05 <= vr <= 0.95:
                            valid_samples.append([target, (hr, vr)])

            if len(valid_samples) < 9:
                print(f"[WARN] 有效样本不足，仅{len(valid_samples)}个，需至少9个")
                self.calibrated_ratio = False
                return

            self.calibration_ratios = valid_samples  # 保留有效样本

            # 2. 确保target_point是归一化坐标（0~1）
            targets = np.array([pair[0] for pair in self.calibration_ratios])
            ratios = np.array([pair[1] for pair in self.calibration_ratios])

            # 强制归一化（防止原始目标坐标未处理）
            screen_w, screen_h = get_screen_resolution()
            # 检查是否已经是归一化坐标（通过检查最大值是否小于等于1）
            if np.max(targets[:, 0]) > 1 or np.max(targets[:, 1]) > 1:
                targets[:, 0] = np.clip(targets[:, 0] / screen_w, 0, 1)  # X归一化
                targets[:, 1] = np.clip(targets[:, 1] / screen_h, 0, 1)  # Y归一化

            # 3. 训练模型
            self.poly_features = PolynomialFeatures(degree=2)  # 降阶到2，避免过拟合
            ratio_poly = self.poly_features.fit_transform(ratios)
            self.regression_model_x = Ridge(alpha=0.5)  # 调整alpha，减少过拟合
            self.regression_model_x.fit(ratio_poly, targets[:, 0])
            self.regression_model_y = Ridge(alpha=0.5)
            self.regression_model_y.fit(ratio_poly, targets[:, 1])

            self.calibrated_ratio = True
            print(f"[INFO] 基础比例校准完成，使用 {len(valid_samples)} 个有效样本")

        except Exception as e:
            print(f"[ERROR] 基础校准模型训练失败: {e}")
            import traceback
            traceback.print_exc()
            self.calibrated_ratio = False

    def get_screen_coords_by_ratio(self, hr, vr, head_pose=None):
        """使用校准数据预测屏幕坐标（增强版，支持头部姿态和分区模型）"""
        # 基础参数检查
        if hr is None or vr is None:
            screen_w, screen_h = get_screen_resolution()
            return (int(hr * screen_w), int(vr * screen_h)) if hr is not None and vr is not None else (
                screen_w // 2, screen_h // 2)

        # 边界检查和修正
        hr = max(0, min(1, hr))
        vr = max(0, min(1, vr))

        screen_w, screen_h = get_screen_resolution()

        # 如果未校准，使用线性映射
        if not self.calibrated_ratio:
            return (int(hr * screen_w), int(vr * screen_h))

        # 如果有分区模型，优先使用
        if hasattr(self, 'zone_models') and head_pose is not None:
            try:
                # 确定当前头部姿态所属区域
                pitch, yaw, roll = head_pose
                current_zone = 'center'  # 默认为中心区域

                for zone_name, zone_range in self.head_pose_zones.items():
                    yaw_min, yaw_max = zone_range['yaw_range']
                    pitch_min, pitch_max = zone_range['pitch_range']

                    if (yaw_min <= yaw <= yaw_max) and (pitch_min <= pitch <= pitch_max):
                        current_zone = zone_name
                        break

                # 使用对应区域的模型
                if current_zone in self.zone_models:
                    zone_model = self.zone_models[current_zone]
                    # 构建特征：HR + VR + 头部姿态
                    feature = np.array([[hr, vr, pitch, yaw, roll]])
                    # 用模型预测归一化坐标
                    x_norm = zone_model['model_x'].predict(feature)[0]
                    y_norm = zone_model['model_y'].predict(feature)[0]
                    # 转换为屏幕坐标
                    screen_x = max(0, min(screen_w, int(x_norm * screen_w)))
                    screen_y = max(0, min(screen_h, int(y_norm * screen_h)))

                    # 自适应修正
                    # 注意：这里需要传入生理特征来进行修正
                    return (screen_x, screen_y)
                else:
                    print(f"[WARN] 未找到区域 '{current_zone}' 的模型，使用全局模型")
            except Exception as e:
                print(f"[WARN] 分区模型预测失败，回退到全局模型: {e}")

        # 如果有头部姿态感知模型，使用全局模型
        if (hasattr(self, 'regression_model_x') and hasattr(self, 'regression_model_y') and
                self.regression_model_x is not None and self.regression_model_y is not None and
                head_pose is not None):
            try:
                # 构建特征：HR + VR + 头部姿态
                feature = np.array([[hr, vr, head_pose[0] if head_pose else 0,
                                     head_pose[1] if head_pose else 0,
                                     head_pose[2] if head_pose else 0]])
                # 用模型预测归一化坐标
                x_norm = self.regression_model_x.predict(feature)[0]
                y_norm = self.regression_model_y.predict(feature)[0]
                # 转换为屏幕坐标
                screen_x = max(0, min(screen_w, int(x_norm * screen_w)))
                screen_y = max(0, min(screen_h, int(y_norm * screen_h)))

                # 自适应修正
                # 注意：这里需要传入生理特征来进行修正
                return (screen_x, screen_y)
            except Exception as e:
                print(f"[WARN] 头部姿态模型预测失败，回退到基础模型: {e}")

        # 尝试使用主回归模型
        primary_result = self._predict_with_primary_model(hr, vr)
        if primary_result is not None:
            return primary_result

        # 尝试使用备用模型
        fallback_result = self._predict_with_fallback_model(hr, vr)
        if fallback_result is not None:
            return fallback_result

        # 最后的回退方案
        return (int(hr * screen_w), int(vr * screen_h))

    def _predict_with_primary_model(self, hr, vr):
        """使用主回归模型预测"""
        try:
            if self.regression_model_x and self.regression_model_y and self.poly_features:
                current_ratio_poly = self.poly_features.transform([[hr, vr]])
                x_pred = self.regression_model_x.predict(current_ratio_poly)[0]
                y_pred = self.regression_model_y.predict(current_ratio_poly)[0]

                # 输出边界检查
                screen_w, screen_h = get_screen_resolution()
                screen_x = max(0, min(screen_w, int(x_pred * screen_w)))
                screen_y = max(0, min(screen_h, int(y_pred * screen_h)))

                # 合理性检查
                if self._is_prediction_reasonable(hr, vr, screen_x, screen_y):
                    return (screen_x, screen_y)
        except Exception as e:
            print(f"[WARN] 主模型预测失败: {e}")
        return None

    def _predict_with_fallback_model(self, hr, vr):
        """使用备用模型预测（如移动平均或最近邻）"""
        try:
            if hasattr(self, 'calibration_ratios') and len(self.calibration_ratios) > 0:
                # 使用最近邻方法作为备用
                targets = np.array([pair[0] for pair in self.calibration_ratios])
                ratios = np.array([pair[1] for pair in self.calibration_ratios])

                # 计算与已知样本的距离
                distances = np.sqrt((ratios[:, 0] - hr) ** 2 + (ratios[:, 1] - vr) ** 2)
                closest_idx = np.argmin(distances)

                # 如果最近邻足够近，使用其目标坐标
                if distances[closest_idx] < 0.1:  # 阈值可调
                    target = targets[closest_idx]
                    screen_w, screen_h = get_screen_resolution()
                    return (int(target[0] * screen_w), int(target[1] * screen_h))
        except Exception as e:
            print(f"[WARN] 备用模型预测失败: {e}")
        return None

    def _is_prediction_reasonable(self, hr, vr, screen_x, screen_y):
        """检查预测结果是否合理"""
        screen_w, screen_h = get_screen_resolution()

        # 检查是否在屏幕范围内（允许一定超出）
        if not (0 <= screen_x <= screen_w and 0 <= screen_y <= screen_h):
            # 轻微超出可以接受，严重超出则不合理
            if not (-screen_w * 0.1 <= screen_x <= screen_w * 1.1 and -screen_h * 0.1 <= screen_y <= screen_h * 1.1):
                return False

        # 可以添加更多合理性检查，如变化率限制等
        return True

    def update_calibration_dynamically(self, new_target_point, new_eye_ratio):
        """动态更新校准数据"""
        try:
            # 添加新样本
            self.calibration_ratios.append([new_target_point, new_eye_ratio])

            # 如果样本足够多，重新训练模型
            if len(self.calibration_ratios) >= 15:  # 动态更新阈值
                self.finalize_ratio_calibration()
                print("[INFO] 动态校准模型已更新")
        except Exception as e:
            print(f"[ERROR] 动态校准更新失败: {e}")

    def _assign_head_pose_zone(self, yaw, pitch):
        """
        根据yaw/pitch确定头部姿态区域
        """
        for zone_name, zone_range in self.head_pose_zones.items():
            yaw_min, yaw_max = zone_range['yaw_range']
            pitch_min, pitch_max = zone_range['pitch_range']
            if yaw_min <= yaw <= yaw_max and pitch_min <= pitch <= pitch_max:
                return zone_name
        return None  # 超出定义范围的姿态

    def add_head_pose_zone_sample(self, gaze_data, screen_coord):
        """
        按头部姿态分配区域，存入对应样本池
        gaze_data: 含 head_pitch/head_yaw 的眼动数据
        screen_coord: 目标屏幕坐标 (target_x, target_y)
        """
        head_pitch = gaze_data.get("head_pitch", 0)
        head_yaw = gaze_data.get("head_yaw", 0)
        head_pose_zone = self._assign_head_pose_zone(head_yaw, head_pitch)

        if head_pose_zone:
            self.head_pose_zone_samples[head_pose_zone].append((gaze_data, screen_coord))
            print(f"[DEBUG] 样本添加到头部姿态区域: {head_pose_zone}")

    def train_head_pose_zone_models(self, n_estimators=100, random_state=42):
        """
        分头部姿态区域训练随机森林模型（增强版，包含超参数自适应）
        """
        from sklearn.ensemble import RandomForestRegressor
        import numpy as np

        # 首先进行自适应区域划分（如果提供了校准样本）
        if hasattr(self, 'calibration_samples') and len(self.calibration_samples) > 0:
            self.adaptive_head_pose_zone_calibration(self.calibration_samples)

        # 执行样本增强，为样本不足的区域生成插值样本
        self.adaptive_sample_augmentation()

        # 遍历每个头部姿态区域，训练专属模型
        trained_zones = []
        for zone_name, samples in self.head_pose_zone_samples.items():
            sample_count = len(samples)

            # 即使样本较少也尝试训练，但我们根据样本量调整模型复杂度
            if sample_count < 3:  # 最小样本要求
                print(f"[WARN] 头部姿态区域 {zone_name} 样本量不足（{sample_count}个），跳过训练")
                continue

            # 1. 根据样本量动态调整模型超参数（避免过拟合/欠拟合）
            if sample_count < 10:
                # 样本量少时减小模型复杂度，避免过拟合
                zone_n_estimators = 50  # 减少树数量
                zone_max_depth = 5  # 降低树深度
                print(f"[INFO] 区域 {zone_name} 样本较少 ({sample_count})，使用简化模型参数")
            elif sample_count > 50:
                # 样本量多时增加模型复杂度，提升拟合能力
                zone_n_estimators = 150
                zone_max_depth = 15
                print(f"[INFO] 区域 {zone_name} 样本充足 ({sample_count})，使用复杂模型参数")
            else:
                # 中等样本量使用默认参数
                zone_n_estimators = 100
                zone_max_depth = 10
                print(f"[INFO] 区域 {zone_name} 样本适中 ({sample_count})，使用标准模型参数")

            # 拆分特征（X）和标签（y：屏幕坐标）
            X = []
            y_x = []
            y_y = []

            for gaze_data, screen_coord in samples:
                # 提取特征：HR, VR, 头部姿态等
                hr = gaze_data.get('hr', 0.5)
                vr = gaze_data.get('vr', 0.5)
                head_pitch = gaze_data.get('head_pitch', 0)
                head_yaw = gaze_data.get('head_yaw', 0)
                head_roll = gaze_data.get('head_roll', 0)

                # 构建特征向量
                features = [hr, vr, head_pitch, head_yaw, head_roll]
                X.append(features)
                y_x.append(screen_coord[0])  # 目标screen_x
                y_y.append(screen_coord[1])  # 目标screen_y

            # 转换为numpy数组
            X_train = np.array(X)
            y_x_train = np.array(y_x)
            y_y_train = np.array(y_y)

            # 训练随机森林（x/y坐标分开预测）
            model_x = RandomForestRegressor(
                n_estimators=zone_n_estimators,
                max_depth=zone_max_depth,
                random_state=random_state,
                min_samples_split=max(2, sample_count // 10),  # 根据样本量调整最小分割样本数
                min_samples_leaf=max(1, sample_count // 20)  # 根据样本量调整叶子节点最小样本数
            )
            model_x.fit(X_train, y_x_train)

            model_y = RandomForestRegressor(
                n_estimators=zone_n_estimators,
                max_depth=zone_max_depth,
                random_state=random_state,
                min_samples_split=max(2, sample_count // 10),
                min_samples_leaf=max(1, sample_count // 20)
            )
            model_y.fit(X_train, y_y_train)

            # 保存模型到区域模型字典
            self.head_pose_zone_models[zone_name] = {"model_x": model_x, "model_y": model_y}
            trained_zones.append(zone_name)

            # 计算该区域模型的误差
            y_x_pred = model_x.predict(X_train)
            y_y_pred = model_y.predict(X_train)

            # 计算误差波动率σ_e
            errors = [np.sqrt((x - xp) ** 2 + (y - yp) ** 2) for (x, y), (xp, yp) in
                      zip(zip(y_x_train, y_y_train), zip(y_x_pred, y_y_pred))]
            mean_error = np.mean(errors)
            error_std = np.std(errors)
            error_volatility = error_std / mean_error if mean_error > 0 else 0  # 误差波动率σ_e

            print(f"[INFO] 头部姿态区域 {zone_name} 模型训练完成：")
            print(f"       - 样本量：{sample_count} | 平均误差：{mean_error:.1f}px | 误差波动率：{error_volatility:.2f}")
            print(f"       - 模型参数：n_estimators={zone_n_estimators}, max_depth={zone_max_depth}")

        print(f"[INFO] 总共训练了 {len(trained_zones)} 个区域模型: {', '.join(trained_zones)}")

        # 为未训练的区域提供备用方案（使用全局模型或简单映射）
        for zone_name in self.head_pose_zones.keys():
            if zone_name not in trained_zones:
                print(f"[WARN] 区域 {zone_name} 未训练模型，将使用全局模型作为备选")

    def predict_by_head_pose(self, gaze_data):
        """
        根据当前头部姿态选择对应区域模型，预测屏幕坐标（增强版）
        """
        # 获取屏幕分辨率
        screen_w, screen_h = get_screen_resolution()

        # 获取当前头部姿态
        head_pitch = gaze_data.get("head_pitch", 0)
        head_yaw = gaze_data.get("head_yaw", 0)
        head_roll = gaze_data.get("head_roll", 0)

        # 使用自适应区域分配
        target_zone = self.assign_sample_to_zone((head_pitch, head_yaw, head_roll))

        # 提取特征
        hr = gaze_data.get('hr', 0.5)
        vr = gaze_data.get('vr', 0.5)

        features = [hr, vr, head_pitch, head_yaw, head_roll]
        X = np.array(features).reshape(1, -1)

        # 选择模型：优先用区域专属模型
        screen_x, screen_y = None, None

        if target_zone in self.head_pose_zone_models and self.head_pose_zone_models[target_zone] is not None:
            try:
                model_x = self.head_pose_zone_models[target_zone]["model_x"]
                model_y = self.head_pose_zone_models[target_zone]["model_y"]
                screen_x = model_x.predict(X)[0]
                screen_y = model_y.predict(X)[0]
                print(f"[DEBUG] 使用头部姿态区域 {target_zone} 的模型预测: ({screen_x:.1f}, {screen_y:.1f})")
            except Exception as e:
                print(f"[WARN] 区域 {target_zone} 模型预测失败: {e}")

        # 如果区域模型不可用或预测失败，回退到全局模型
        if screen_x is None or screen_y is None:
            if hasattr(self, 'regression_model_x') and self.regression_model_x is not None:
                try:
                    # 使用已有的回归模型
                    screen_x = self.regression_model_x.predict(X)[0]
                    screen_y = self.regression_model_y.predict(X)[0]
                    print(f"[DEBUG] 使用全局回归模型预测: ({screen_x:.1f}, {screen_y:.1f})")
                except Exception as e:
                    print(f"[WARN] 全局模型预测失败: {e}")

        # 如果所有模型都失败，使用基础线性映射
        if screen_x is None or screen_y is None:
            # 最基础的线性映射
            screen_x = hr * screen_w
            screen_y = vr * screen_h
            print(f"[DEBUG] 使用基础线性映射预测: ({screen_x:.1f}, {screen_y:.1f})")

        # 确保坐标在屏幕范围内
        screen_x = max(0, min(screen_w, screen_x))
        screen_y = max(0, min(screen_h, screen_y))

        return int(screen_x), int(screen_y)

    def get_adjacent_zones(self, zone):
        """
        获取指定区域的相邻区域列表

        参数:
            zone: 区域名称字符串

        返回:
            相邻区域名称列表
        """
        # 定义相邻区域映射（示例：中心相邻左/右/上/下）
        adjacent = {
            "center": ["left", "right", "up", "down"],
            "left": ["center", "up", "down"],
            "right": ["center", "up", "down"],
            "up": ["center", "left", "right"],
            "down": ["center", "left", "right"]
        }
        return adjacent.get(zone, [])

    def interpolate_samples(self, src_samples, n):
        """
        从源样本中插值生成n个新样本（保持特征分布）

        参数:
            src_samples: 源样本列表，可以是字典或元组格式
            n: 需要生成的插值样本数量

        返回:
            interpolated_samples: 插值生成的样本列表
        """
        if len(src_samples) < 2:
            return []

        interpolated_samples = []

        for _ in range(n):
            # 随机选择两个不相同的样本进行插值
            idx1, idx2 = np.random.choice(len(src_samples), 2, replace=False)
            sample1 = src_samples[idx1]
            sample2 = src_samples[idx2]

            # 创建插值样本
            interpolated_sample = {}

            # 处理字典格式样本
            if isinstance(sample1, dict) and isinstance(sample2, dict):
                for key in sample1.keys():
                    if key in sample2 and isinstance(sample1[key], (int, float)) and isinstance(sample2[key],
                                                                                                (int, float)):
                        # 对数值特征进行线性插值
                        interpolated_sample[key] = (sample1[key] + sample2[key]) / 2
                    else:
                        # 非数值特征取第一个样本的值
                        interpolated_sample[key] = sample1[key] if key in sample1 else sample2.get(key)
            else:
                # 处理元组格式样本
                try:
                    # 假设样本格式为 (target_point, eye_ratio, head_pose)
                    target_point1, eye_ratio1, head_pose1 = sample1
                    target_point2, eye_ratio2, head_pose2 = sample2

                    # 插值目标点
                    interp_target = (
                        (target_point1[0] + target_point2[0]) / 2,
                        (target_point1[1] + target_point2[1]) / 2
                    )

                    # 插值眼动比率
                    interp_ratio = (
                        (eye_ratio1[0] + eye_ratio2[0]) / 2,
                        (eye_ratio1[1] + eye_ratio2[1]) / 2
                    )

                    # 插值头部姿态
                    interp_head_pose = tuple(
                        (head_pose1[i] + head_pose2[i]) / 2 for i in range(len(head_pose1))
                    )

                    interpolated_sample = (interp_target, interp_ratio, interp_head_pose)
                except Exception as e:
                    # 如果解析失败，复制第一个样本
                    interpolated_sample = sample1

            interpolated_samples.append(interpolated_sample)

        return interpolated_samples

    def adaptive_sample_augmentation(self):
        """
        自适应样本增强：为样本不足的区域生成插值样本
        """
        # 检查每个区域的样本数量
        for zone_name, samples in self.head_pose_zone_samples.items():
            if len(samples) < 5:
                print(f"[INFO] 区域 '{zone_name}' 样本不足 ({len(samples)} 个)，进行插值补全")

                # 获取相邻区域的样本
                adjacent_zones = self.get_adjacent_zones(zone_name)
                adjacent_samples = []
                for adj_zone in adjacent_zones:
                    if adj_zone in self.head_pose_zone_samples:
                        adjacent_samples.extend(self.head_pose_zone_samples[adj_zone])

                # 生成插值样本
                num_needed = 5 - len(samples)
                if num_needed > 0 and len(adjacent_samples) >= 2:
                    interpolated_samples = self.interpolate_samples(adjacent_samples, num_needed)
                    # 将插值样本添加到当前区域
                    for sample in interpolated_samples:
                        if isinstance(sample, tuple) and len(sample) >= 2:
                            # 对于元组格式样本，需要构造完整的数据结构
                            gaze_data, screen_coord = sample[0] if isinstance(sample[0], (tuple, list)) and len(
                                sample) > 1 else (sample, (0, 0))
                            self.head_pose_zone_samples[zone_name].append((gaze_data, screen_coord))
                        else:
                            # 对于字典格式样本，直接添加
                            screen_coord = (0, 0)  # 默认屏幕坐标
                            self.head_pose_zone_samples[zone_name].append((sample, screen_coord))

                    print(f"[INFO] 为区域 '{zone_name}' 生成 {len(interpolated_samples)} 个插值样本")

    def get_adjacent_zones(self, zone_name):
        """
        获取相邻区域列表
        """
        adjacency_map = {
            'center': ['left', 'right', 'up', 'down'],
            'left': ['center', 'up', 'down'],
            'right': ['center', 'up', 'down'],
            'up': ['center', 'left', 'right'],
            'down': ['center', 'left', 'right']
        }
        return adjacency_map.get(zone_name, ['center'])

    def interpolate_samples(self, adjacent_samples, num_needed):
        """
        通过相邻区域样本插值生成新的样本

        参数:
            adjacent_samples: 相邻区域的样本列表
            num_needed: 需要生成的样本数量

        返回:
            interpolated_samples: 插值生成的样本列表
        """
        if len(adjacent_samples) < 2:
            return []

        interpolated_samples = []

        # 随机选择样本进行插值
        for i in range(num_needed):
            # 随机选择两个样本进行插值
            idx1 = np.random.randint(0, len(adjacent_samples))
            idx2 = np.random.randint(0, len(adjacent_samples))

            sample1 = adjacent_samples[idx1]
            sample2 = adjacent_samples[idx2]

            # 创建插值样本
            interpolated_sample = {}

            # 对数值特征进行线性插值
            if isinstance(sample1, dict) and isinstance(sample2, dict):
                # 处理字典格式样本
                for key in sample1.keys():
                    if key in sample2 and isinstance(sample1[key], (int, float)) and isinstance(sample2[key],
                                                                                                (int, float)):
                        # 线性插值
                        interpolated_sample[key] = (sample1[key] + sample2[key]) / 2
                    else:
                        # 非数值特征取第一个样本的值
                        interpolated_sample[key] = sample1[key] if key in sample1 else sample2.get(key)
            else:
                # 处理元组格式样本
                try:
                    # 假设样本格式为 (target_point, eye_ratio, head_pose)
                    target_point1, eye_ratio1, head_pose1 = sample1
                    target_point2, eye_ratio2, head_pose2 = sample2

                    # 插值目标点
                    interp_target = (
                        (target_point1[0] + target_point2[0]) / 2,
                        (target_point1[1] + target_point2[1]) / 2
                    )

                    # 插值眼动比率
                    interp_ratio = (
                        (eye_ratio1[0] + eye_ratio2[0]) / 2,
                        (eye_ratio1[1] + eye_ratio2[1]) / 2
                    )

                    # 插值头部姿态
                    interp_head_pose = tuple(
                        (head_pose1[i] + head_pose2[i]) / 2 for i in range(len(head_pose1))
                    )

                    interpolated_sample = (interp_target, interp_ratio, interp_head_pose)
                except Exception as e:
                    # 如果解析失败，复制第一个样本
                    interpolated_sample = sample1

            interpolated_samples.append(interpolated_sample)

        return interpolated_samples

    def adaptive_sample_augmentation(self):
        """
        自适应样本增强：为样本不足的区域生成插值样本
        """
        # 检查每个区域的样本数量
        for zone_name, samples in self.head_pose_zone_samples.items():
            if len(samples) < 5:
                print(f"[INFO] 区域 '{zone_name}' 样本不足 ({len(samples)} 个)，进行插值补全")

                # 获取相邻区域的样本
                adjacent_zones = self.get_adjacent_zones(zone_name)
                adjacent_samples = []
                for adj_zone in adjacent_zones:
                    if adj_zone in self.head_pose_zone_samples:
                        adjacent_samples.extend(self.head_pose_zone_samples[adj_zone])

                # 生成插值样本
                num_needed = 5 - len(samples)
                if num_needed > 0 and len(adjacent_samples) >= 2:
                    interpolated_samples = self.interpolate_samples(adjacent_samples, num_needed)
                    # 将插值样本添加到当前区域
                    for sample in interpolated_samples:
                        if isinstance(sample, tuple) and len(sample) >= 2:
                            # 对于元组格式样本，需要构造完整的数据结构
                            gaze_data, screen_coord = sample[0] if isinstance(sample[0], (tuple, list)) and len(
                                sample) > 1 else (sample, (0, 0))
                            self.head_pose_zone_samples[zone_name].append((gaze_data, screen_coord))
                        else:
                            # 对于字典格式样本，直接添加
                            screen_coord = (0, 0)  # 默认屏幕坐标
                            self.head_pose_zone_samples[zone_name].append((sample, screen_coord))

                    print(f"[INFO] 为区域 '{zone_name}' 生成 {len(interpolated_samples)} 个插值样本")

    def adaptive_head_pose_zone_calibration(self, calibration_samples):
        """
        基于用户实际头部运动范围自适应划分区域

        参数:
            calibration_samples: 校准样本列表
        """
        if len(calibration_samples) < 10:
            print("[WARN] 校准样本不足，使用默认区域划分")
            return

        # 提取所有头部姿态数据
        yaw_values = []
        pitch_values = []

        for sample in calibration_samples:
            if isinstance(sample, dict):
                # 处理字典格式样本
                head_pose = sample.get('head_pose')
                if head_pose and len(head_pose) >= 2:
                    pitch_values.append(head_pose[0])
                    yaw_values.append(head_pose[1])
            elif isinstance(sample, (list, tuple)) and len(sample) >= 3:
                # 处理元组格式样本
                head_pose = sample[2]  # 假设第三个元素是头部姿态
                if head_pose and len(head_pose) >= 2:
                    pitch_values.append(head_pose[0])
                    yaw_values.append(head_pose[1])

        if len(yaw_values) < 5 or len(pitch_values) < 5:
            print("[WARN] 头部姿态数据不足，使用默认区域划分")
            return

        # 计算统计值
        yaw_min, yaw_max = np.min(yaw_values), np.max(yaw_values)
        pitch_min, pitch_max = np.min(pitch_values), np.max(pitch_values)

        # 添加边界扩展，避免极端值影响
        yaw_range = yaw_max - yaw_min
        pitch_range = pitch_max - pitch_min
        yaw_min -= yaw_range * 0.05  # 扩展5%
        yaw_max += yaw_range * 0.05
        pitch_min -= pitch_range * 0.05
        pitch_max += pitch_range * 0.05

        # 计算三分位点
        yaw_third = (yaw_max - yaw_min) / 3
        pitch_third = (pitch_max - pitch_min) / 3

        # 自适应划分5个区域
        self.head_pose_zones = {
            'center': {
                'yaw_range': (yaw_min + yaw_third, yaw_max - yaw_third),
                'pitch_range': (pitch_min + pitch_third, pitch_max - pitch_third)
            },
            'left': {
                'yaw_range': (yaw_min, yaw_min + yaw_third),
                'pitch_range': (pitch_min, pitch_max)
            },
            'right': {
                'yaw_range': (yaw_max - yaw_third, yaw_max),
                'pitch_range': (pitch_min, pitch_max)
            },
            'up': {
                'yaw_range': (yaw_min, yaw_max),
                'pitch_range': (pitch_max - pitch_third, pitch_max)
            },
            'down': {
                'yaw_range': (yaw_min, yaw_max),
                'pitch_range': (pitch_min, pitch_min + pitch_third)
            }
        }

        print(f"[INFO] 自适应区域划分完成:")
        print(f"       Yaw范围: [{yaw_min:.2f}, {yaw_max:.2f}]")
        print(f"       Pitch范围: [{pitch_min:.2f}, {pitch_max:.2f}]")
        for zone, ranges in self.head_pose_zones.items():
            print(f"       {zone}: Yaw{ranges['yaw_range']}, Pitch{ranges['pitch_range']}")

    # 在 DirectionCalibrator 类中，大约在 1000-1200 行左右的位置
    # 替换原有的 adaptive_head_pose_zone_calibration 方法
    def calibrate_head_pose_zones(self, user_calib_samples):
        """
        基于用户校准样本自适应划分头部姿态区域

        参数:
            user_calib_samples: 用户校准样本列表，每个样本包含头部姿态信息
        """
        all_yaw = [s["head_yaw"] for s in user_calib_samples]
        all_pitch = [s["head_pitch"] for s in user_calib_samples]
        yaw_min, yaw_max = min(all_yaw), max(all_yaw)
        pitch_min, pitch_max = min(all_pitch), max(all_pitch)

        # 三等分划分区域
        yaw_1 = yaw_min + (yaw_max - yaw_min) / 3
        yaw_2 = yaw_max - (yaw_max - yaw_min) / 3
        pitch_1 = pitch_min + (pitch_max - pitch_min) / 3
        pitch_2 = pitch_max - (pitch_max - pitch_min) / 3

        # 保持与现有代码一致的数据结构格式
        self.head_pose_zones = {
            "center": {"yaw_range": (yaw_1, yaw_2), "pitch_range": (pitch_1, pitch_2)},
            "left": {"yaw_range": (yaw_min, yaw_1), "pitch_range": (pitch_min, pitch_max)},
            "right": {"yaw_range": (yaw_2, yaw_max), "pitch_range": (pitch_min, pitch_max)},
            "up": {"yaw_range": (yaw_min, yaw_max), "pitch_range": (pitch_min, pitch_1)},
            "down": {"yaw_range": (yaw_min, yaw_max), "pitch_range": (pitch_2, pitch_max)}
        }

    def assign_sample_to_zone(self, head_pose):
        """
        将样本分配到最适合的区域

        参数:
            head_pose: 头部姿态 (pitch, yaw, roll)

        返回:
            zone_name: 区域名称
        """
        if not head_pose or len(head_pose) < 2:
            return 'center'

        pitch, yaw = head_pose[0], head_pose[1]

        # 计算样本到各个区域中心的距离，分配到最近的区域
        min_distance = float('inf')
        best_zone = 'center'

        for zone_name, zone_ranges in self.head_pose_zones.items():
            yaw_range = zone_ranges['yaw_range']
            pitch_range = zone_ranges['pitch_range']

            # 计算区域中心
            zone_yaw_center = (yaw_range[0] + yaw_range[1]) / 2
            zone_pitch_center = (pitch_range[0] + pitch_range[1]) / 2

            # 计算欧几里得距离
            distance = np.sqrt((yaw - zone_yaw_center) ** 2 + (pitch - zone_pitch_center) ** 2)

            if distance < min_distance:
                min_distance = distance
                best_zone = zone_name

        return best_zone


def train_advanced_mapping_model(self):
    """
    训练更先进的映射模型，结合多种方法和特征集
    """
    if len(self.calibration_samples) < 5:
        print("[WARN] 样本不足，无法训练高级映射模型")
        self.has_advanced_model = False
        self.advanced_model_trained = False
        return

    try:
        # 导入所需模块
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler

        # 准备训练数据 - 方法1: 全特征集
        X_full = []  # 特征: [hr, vr, head_pitch, head_yaw, head_roll, gaze_vx, gaze_vy, gaze_vz]
        X_simple = []  # 特征: [hr, yaw, pitch] 简化特征集
        y = []  # 目标: [target_x, target_y]
        y_x = []  # X坐标目标
        y_y = []  # Y坐标目标

        for sample in self.calibration_samples:
            # 确保样本包含必要数据
            if 'hr_vr' not in sample or 'target_point' not in sample:
                continue

            # 提取完整特征集
            features_full = [
                sample['hr_vr'][0],  # hr
                sample['hr_vr'][1],  # vr
                sample.get('head_pose', [0, 0, 0])[0] if sample.get('head_pose', [0, 0, 0])[0] is not None else 0,
                # pitch
                sample.get('head_pose', [0, 0, 0])[1] if sample.get('head_pose', [0, 0, 0])[1] is not None else 0,
                # yaw
                sample.get('head_pose', [0, 0, 0])[2] if sample.get('head_pose', [0, 0, 0])[2] is not None else 0,
                # roll
                sample.get('gaze_vector', [0, 0, 0])[0] if sample.get('gaze_vector', [0, 0, 0])[0] is not None else 0,
                # gaze_vx
                sample.get('gaze_vector', [0, 0, 0])[1] if sample.get('gaze_vector', [0, 0, 0])[1] is not None else 0,
                # gaze_vy
                sample.get('gaze_vector', [0, 0, 0])[2] if sample.get('gaze_vector', [0, 0, 0])[2] is not None else 0
                # gaze_vz
            ]

            # 提取简化特征集
            hr, vr = sample['hr_vr']
            pitch, yaw, roll = sample.get('head_pose', [0, 0, 0])
            features_simple = [hr, yaw, pitch]

            # 提取目标值
            target_x, target_y = sample['target_point']

            X_full.append(features_full)
            X_simple.append(features_simple)
            y.append(sample['target_point'])
            y_x.append(target_x)
            y_y.append(target_y)

        if len(X_full) < 5:
            print("[WARN] 有效样本不足，无法训练模型")
            self.has_advanced_model = False
            self.advanced_model_trained = False
            return

        X_full = np.array(X_full)
        X_simple = np.array(X_simple)
        y = np.array(y)
        y_x = np.array(y_x)
        y_y = np.array(y_y)

        # 方法1: 使用随机森林和全特征集
        # 数据标准化
        self.scaler = StandardScaler()
        X_full_scaled = self.scaler.fit_transform(X_full)

        # 使用随机森林
        self.advanced_mapping_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        # 训练模型
        self.advanced_mapping_model.fit(X_full_scaled, y)
        self.has_advanced_model = True

        # 方法2: 使用线性回归和简化特征集
        # 训练X坐标模型
        self.x_model = LinearRegression()
        self.x_model.fit(X_simple, y_x)
        # 训练Y坐标模型
        self.y_model = LinearRegression()
        self.y_model.fit(X_simple, y_y)
        self.advanced_model_trained = True

        print(f"[INFO] 高级映射模型训练完成，使用 {len(X_full)} 个样本")
        print("[INFO] 提供两种模型: 随机森林(全特征)和线性回归(简化特征)")

    except Exception as e:
        print(f"[ERROR] 训练高级映射模型失败: {e}")
        import traceback
        traceback.print_exc()
        self.has_advanced_model = False
        self.advanced_model_trained = False


def predict_screen_coordinates_advanced(self, gaze_data):
    """
    使用高级模型预测屏幕坐标（融合生理特征版本）
    """
    try:
        # 获取生理特征
        physiological_features = gaze_data.get('physiological_features', {})
        pupil_ellipticity = physiological_features.get('pupil_ellipticity_left', 1.0) \
            if gaze_data.get('left_pupil') else physiological_features.get('pupil_ellipticity_right', 1.0)
        eyelid_occlusion_rate = physiological_features.get('eyelid_occlusion_left', 0.0) \
            if gaze_data.get('left_pupil') else physiological_features.get('eyelid_occlusion_right', 0.0)
        eye_dominance = physiological_features.get('eye_dominance', 0.0)

        # 获取基础数据
        hr = gaze_data.get('hr', 0.5)
        vr = gaze_data.get('vr', 0.5)
        head_pose = gaze_data.get('head_pose', (0, 0, 0))

        if head_pose and len(head_pose) >= 3:
            pitch, yaw, roll = head_pose
        else:
            pitch, yaw, roll = 0, 0, 0

        # 自适应调整HR/VR值（基于生理特征）
        adjusted_hr = self._adjust_hr_for_ellipticity(hr, pupil_ellipticity)
        adjusted_vr = self._adjust_vr_for_occlusion(vr, eyelid_occlusion_rate)

        # 添加调试日志 - 显示调整前后的值
        print(f"[DEBUG] 预测输入特征 - HR: {hr:.3f}->{adjusted_hr:.3f}, VR: {vr:.3f}->{adjusted_vr:.3f}, "
              f"Pitch: {pitch:.3f}, Yaw: {yaw:.3f}, Roll: {roll:.3f}")
        print(
            f"[DEBUG] 生理特征 - 椭圆度: {pupil_ellipticity:.3f}, 遮挡率: {eyelid_occlusion_rate:.3f}, 优势度: {eye_dominance:.3f}")

        # 计算生理特征影响因子 φ = 1 + k1(1 - e) - k2*o
        k1 = 0.3  # 瞳孔椭圆度权重
        k2 = 0.2  # 眼睑遮挡权重
        phi = 1 + k1 * (1 - pupil_ellipticity) - k2 * eyelid_occlusion_rate

        # 1. 首先尝试使用训练好的高级模型（融入生理特征）
        has_advanced_model = hasattr(self, 'has_advanced_model') and self.has_advanced_model
        has_scaler = hasattr(self, 'scaler')
        has_model = hasattr(self, 'advanced_mapping_model')

        if has_advanced_model and has_scaler and has_model:
            # 准备特征向量（包含生理特征）
            features = np.array([[[
                adjusted_hr,
                adjusted_vr,
                pitch,
                yaw,
                roll,
                pupil_ellipticity,
                eyelid_occlusion_rate,
                eye_dominance,
                gaze_data.get('gaze_vx', 0),
                gaze_data.get('gaze_vy', 0),
                gaze_data.get('gaze_vz', 1)
            ]]])

            # 标准化特征
            features_scaled = self.scaler.transform(features)

            # 预测归一化的屏幕坐标
            prediction = self.advanced_mapping_model.predict(features_scaled)[0]

            # 应用生理特征影响因子进行修正
            screen_w, screen_h = get_screen_resolution()
            screen_x = int(max(0, min(screen_w, prediction[0] * screen_w * phi)))
            screen_y = int(max(0, min(screen_h, prediction[1] * screen_h * phi)))

            return (screen_x, screen_y)

        # 2. 如果没有高级模型，尝试使用线性回归模型
        elif hasattr(self, 'x_model') and hasattr(self, 'y_model') and hasattr(self,
                                                                               'advanced_model_trained') and self.advanced_model_trained:
            # 预测归一化坐标
            x_norm = self.x_model.predict([[adjusted_hr, yaw, pitch, pupil_ellipticity, eyelid_occlusion_rate]])[0]
            y_norm = self.y_model.predict([[adjusted_vr, yaw, pitch, pupil_ellipticity, eyelid_occlusion_rate]])[0]

            # 应用生理特征影响因子进行修正并转换为屏幕像素坐标
            screen_w, screen_h = get_screen_resolution()
            screen_x = int(x_norm * screen_w * phi)
            screen_y = int(y_norm * screen_h * phi)
            return (screen_x, screen_y)

        # 3. 使用固定系数方式计算（融入生理特征修正）
        else:
            # 基于校准样本训练得到的系数（固定值）
            # X坐标 = a1*HR + a2*Yaw + a3
            screen_x = 1720 * adjusted_hr + 10 * yaw - 500
            # Y坐标 = b1*VR + b2*Pitch + b3
            screen_y = 1080 * adjusted_vr - 8 * pitch + 200

            # 应用生理特征影响因子进行修正
            screen_x = int(screen_x * phi)
            screen_y = int(screen_y * phi)

            # 限制在屏幕范围内
            screen_w, screen_h = get_screen_resolution()
            screen_x = max(0, min(screen_x, screen_w))
            screen_y = max(0, min(screen_y, screen_h))

            return (screen_x, screen_y)

    except Exception as e:
        print(f"[ERROR] 高级模型预测失败: {e}")
        # 最后的回退方案：使用简单映射
        return self.get_screen_coords_by_ratio(
            gaze_data.get('hr'),
            gaze_data.get('vr'),
            gaze_data.get('head_pose')
        )



def _adjust_hr_for_ellipticity(self, hr, ellipticity):
    """
    根据瞳孔椭圆度调整水平比例

    Args:
        hr: 原始水平比例
        ellipticity: 瞳孔椭圆度 (0-1, 1表示正圆)
    """
    # 对于椭圆瞳孔，降低HR对X坐标的影响
    if ellipticity < 0.8:
        # 椭圆度越小，HR权重越低
        adjustment_factor = 1.0 - (0.8 - ellipticity) * 0.3
        # 调整HR使其更接近中心值(0.5)
        adjusted_hr = 0.5 + (hr - 0.5) * adjustment_factor
        adjusted_hr = max(0.01, min(0.99, adjusted_hr))

        # 添加调试日志 - 椭圆瞳孔补偿日志
        print(f"[DEBUG] 椭圆度补偿 - 椭圆度={ellipticity:.3f}, 调整因子={adjustment_factor:.3f}, "
              f"原始HR={hr:.3f}, 调整后HR={adjusted_hr:.3f}")
        return adjusted_hr
    return hr


def _adjust_vr_for_occlusion(self, vr, occlusion_rate):
    """
    根据眼睑遮挡率调整垂直比例

    Args:
        vr: 原始垂直比例
        occlusion_rate: 眼睑遮挡率 (0-1)
    """
    # 高遮挡率时，用动态补偿修正VR
    if occlusion_rate > 0.3:
        # 根据遮挡程度调整VR
        compensation = occlusion_rate * 0.15  # 补偿系数
        if vr < 0.5:  # 上看
            adjusted_vr = vr + compensation
        else:  # 下看
            adjusted_vr = vr - compensation
        adjusted_vr = max(0.01, min(0.99, adjusted_vr))

        # 添加调试日志 - 眼睑遮挡补偿日志
        print(f"[DEBUG] 遮挡率补偿 - 遮挡率={occlusion_rate:.3f}, 补偿值={compensation:.3f}, "
              f"原始VR={vr:.3f}, 调整后VR={adjusted_vr:.3f}")
        return adjusted_vr
    return vr


class Calibrator:
    """
    校准器适配器，统一接口
    """

    def __init__(self):
        self.calibrator = DirectionCalibrator()

    def start_calibration(self, screen_w, screen_h):
        pass

    def finalize_calibration(self):
        self.calibrator.finalize_ratio_calibration()

    def get_screen_coords(self, hr, vr):
        return self.calibrator.get_screen_coords_by_ratio(hr, vr)

    def add_sample(self, hr, vr, target_point=None):
        if target_point:
            self.calibrator.calibration_ratios.append([target_point, (hr, vr)])
