import numpy as np
import copy


class GazeDirectionDetector:
    """
    注视方向判断类，专门处理注视方向的3D向量计算
    包含头部滚转校正功能
    与GazeTracking协同工作但功能解耦，专注于对这些数据进行高级分析，包括滚转校正和3D注视方向计算

    头部滚转校正（使用原始眼部数据）
    视线方向计算（水平和垂直角度）
    3D注视向量生成
    头部姿态与眼部偏移的融合计算
    高级视线分析（如注视点预测）
    """

    def __init__(self, calibration_params=None):
        """
        初始化注视方向检测器

        参数:
            calibration_params (dict): 用户校准参数，包含:
                horizontal_sensitivity: 水平敏感度系数
                vertical_sensitivity: 垂直敏感度系数
                head_yaw_factor: 头部偏航角权重
                head_pitch_factor: 头部俯仰角权重
        """
        # 默认校准参数
        # self.params = {
        #     # 'horizontal_sensitivity': 25.0,  # 水平敏感度(度/单位比率)
        #     # 'vertical_sensitivity': 20.0,  # 垂直敏感度(度/单位比率)
        #     # 'head_yaw_factor': 0.7,  # 头部偏航角权重
        #     # 'head_pitch_factor': 0.6,  # 头部俯仰角权重
        #
        #     'horizontal_sensitivity': 35.0,  # 水平灵敏度（范围20-40）
        #     'vertical_sensitivity': 30.0,  # 垂直灵敏度（范围15-35）
        #     'head_yaw_factor': 0.5,  # 头部偏航角权重（范围0.3-0.8）
        #     'head_pitch_factor': 0.2,  # 头部俯仰角权重（范围0.3-0.7）
        #     'pupil_weight': 1.2,  # 降低瞳孔移动权重（范围1.0-1.8）
        #     'roll_threshold': 10.0 # 滚转校正阈值(度)（范围10-20）
        # }

        # 调整参数以提高灵敏度
        # self.params = {
        #     'horizontal_sensitivity': 45.0,  # 增加水平灵敏度
        #     'vertical_sensitivity': 40.0,  # 增加垂直灵敏度
        #     'head_yaw_factor': 0.3,  # 降低头部偏航影响
        #     'head_pitch_factor': 0.1,  # 降低头部俯仰影响
        #     'pupil_weight': 1.5,  # 调整瞳孔权重
        #     'roll_threshold': 8.0  # 调整滚转校正阈值
        # }

        # 默认校准参数
        self.params = {
            'horizontal_sensitivity': 45.0,  # 水平灵敏度
            'vertical_sensitivity': 40.0,  # 垂直灵敏度
            'theta_yaw': 15.0,  # yaw动态权重阈值（新增）
            'theta_pitch': 15.0,  # pitch动态权重阈值（新增）
            'roll_threshold': 8.0,  # 滚转校正阈值
            'ellipticity_weight': 0.3,  # 瞳孔椭圆度权重（新增）
            'occlusion_weight': 0.2,  # 眼睑遮挡权重（新增）
            'dominance_weight': 0.1  # 眼优势权重（新增）
        }

        # 合并用户自定义参数
        if calibration_params:
            self.params.update(calibration_params)

        # 添加类变量
        self.angle_history = []
        self.vector_history = []

    # 在 gaze_direction.py 的 GazeDirectionDetector 类中添加方法
    def get_gaze_angles(self, gaze_data):
        """
        获取水平和垂直角度观测值
        返回: (h_angle, v_angle)
        """
        # 应用滚转校正
        head_pose = gaze_data.get("head_pose")
        if head_pose is None:
            return 0, 0

        _, _, roll = head_pose
        corrected_data = self.apply_roll_correction(gaze_data, roll)

        # 获取用户自定义敏感度系数（如果存在）
        horizontal_sens = gaze_data.get("user_horizontal_sens", 1.0)
        vertical_sens = gaze_data.get("user_vertical_sens", 1.0)

        # 计算水平和垂直视线角度，并应用系数
        h_angle = self.calc_horizontal_angle(corrected_data) * horizontal_sens
        v_angle = self.calc_vertical_angle(corrected_data) * vertical_sens

        return h_angle, v_angle

    # def detect_gaze_direction(self, gaze_data):
    #     """
    #     综合计算注视方向3D向量
    #     参数:gaze_data (dict): 来自GazeTracking.process_frame()的结构化数据
    #     返回:
    #         tuple: (水平角度, 垂直角度) 单位: 度
    #         np.array: 3D注视方向向量 [x, y, z]
    #     """
    #     # 1. 提取头部姿态数据
    #     head_pose = gaze_data.get("head_pose")
    #     if head_pose is None:
    #         return (0, 0), np.array([0, 0, 1])
    #
    #     pitch, yaw, roll = head_pose
    #
    #     # 2. 应用滚转校正
    #     corrected_data = self.apply_roll_correction(gaze_data, roll)
    #
    #     # 3. 计算水平视线角度
    #     h_angle = self.calc_horizontal_angle(corrected_data)
    #
    #     # 4. 计算垂直视线角度
    #     v_angle = self.calc_vertical_angle(corrected_data)
    #
    #     # 5. 融合头部姿态
    #     # 计算水平视线方向值，平衡头部偏航角（head_yaw）和水平视线角度（h_angle）的影响
    #     # 确保两个权重因子之和为 1
    #     total_horizontal = (
    #             self.params['head_yaw_factor'] * yaw +
    #             (1 - self.params['head_yaw_factor']) * h_angle
    #     )
    #
    #     total_vertical = (
    #             self.params['head_pitch_factor'] * pitch +
    #             (1 - self.params['head_pitch_factor']) * v_angle
    #     )
    #
    #     print("total_horizontal 总水平角度:", total_horizontal)
    #     print("total_vertical 总垂直角度:", total_vertical)
    #
    #     # total_horizontal = yaw + h_angle  # 直接相加
    #     # total_vertical = pitch + v_angle  # 直接相加
    #
    #     # 6. 生成3D注视向量
    #     gaze_vector = self.angles_to_vector(total_horizontal, total_vertical)
    #
    #     # 在返回结果前添加卡尔曼滤波
    #     # 使用历史角度数据对当前角度进行加权平滑处理，减少抖动，提升稳定性。
    #     if len(self.angle_history) > 0:
    #         # 如果历史数据存在，则对当前的水平角度 total_horizontal 进行加权平均
    #         # 这里使用 0.7 的权重给予当前角度，使用 0.3 的权重给予历史最后一个角度
    #         total_horizontal = 0.7 * total_horizontal + 0.3 * self.angle_history[-1][0]
    #         total_vertical = 0.7 * total_vertical + 0.3 * self.angle_history[-1][1]
    #
    #     self.angle_history.append((total_horizontal, total_vertical))
    #     # 数据列表的长度超过 10，移除最早的数据（先进先出）
    #     if len(self.angle_history) > 10:
    #         self.angle_history.pop(0)
    #
    #     return (total_horizontal, total_vertical), gaze_vector

    # 增强 detect_gaze_direction 方法，增强头部姿态补偿
    # 在 gaze_direction.py 的 GazeDirectionDetector 类中修改 detect_gaze_direction 方法
    def detect_gaze_direction(self, gaze_data):
        """
        综合计算注视方向3D向量
        """
        # 1. 提取头部姿态数据
        head_pose = gaze_data.get("head_pose")
        if head_pose is None:
            return (0, 0), np.array([0, 0, 1])

        pitch, yaw, roll = head_pose

        # 2. 应用滚转校正
        corrected_data = self.apply_roll_correction(gaze_data, roll)

        # 3. 计算水平视线角度
        h_angle = self.calc_horizontal_angle(corrected_data)

        # 4. 计算垂直视线角度
        v_angle = self.calc_vertical_angle(corrected_data)

        # 5. 动态权重融合（创新点实现）
        # 获取动态权重参数
        theta_yaw = self.params.get('theta_yaw', 15.0)  # yaw阈值，默认15度
        theta_pitch = self.params.get('theta_pitch', 15.0)  # pitch阈值，默认15度

        # 计算动态权重
        alpha_t = self._dynamic_weight(yaw, theta_yaw)
        beta_t = self._dynamic_weight(pitch, theta_pitch)

        # 动态权重融合
        total_horizontal = alpha_t * yaw + (1 - alpha_t) * h_angle
        total_vertical = beta_t * pitch + (1 - beta_t) * v_angle

        # 6. 融合生理特征进行动态修正
        # 获取生理特征
        physiological_features = gaze_data.get('physiological_features', {})
        pupil_ellipticity = physiological_features.get('pupil_ellipticity_left', 1.0) \
            if gaze_data.get('left_pupil') else physiological_features.get('pupil_ellipticity_right', 1.0)
        eyelid_occlusion = physiological_features.get('eyelid_occlusion_left', 0.0) \
            if gaze_data.get('left_pupil') else physiological_features.get('eyelid_occlusion_right', 0.0)

        # 计算生理特征影响因子 φ = 1 + k1(1 - e) - k2*o
        k1 = self.params.get('ellipticity_weight', 0.3)  # 瞳孔椭圆度权重
        k2 = self.params.get('occlusion_weight', 0.2)  # 眼睑遮挡权重

        phi = 1 + k1 * (1 - pupil_ellipticity) - k2 * eyelid_occlusion

        # 应用生理特征修正
        total_horizontal *= phi
        total_vertical *= phi

        # 7. 生成3D注视向量
        gaze_vector = self.angles_to_vector(total_horizontal, total_vertical)

        # 8. 获取屏幕坐标（新增调用 calculate_screen_coords）
        # 从 gaze_data 中获取 HR/VR
        hr = gaze_data.get("hr", 0.5)
        vr = gaze_data.get("vr", 0.5)

        # 获取屏幕分辨率
        screen_w, screen_h = self._get_screen_resolution()

        # 调用 calculate_screen_coords 计算屏幕坐标
        screen_x, screen_y = self.calculate_screen_coords(hr, vr, yaw, pitch, screen_w, screen_h)

        # 将屏幕坐标添加到 gaze_data 中
        gaze_data["screen_x"] = screen_x
        gaze_data["screen_y"] = screen_y

        # 9. 应用卡尔曼滤波平滑
        if len(self.angle_history) > 0:
            alpha = 0.7
            total_horizontal = alpha * total_horizontal + (1 - alpha) * self.angle_history[-1][0]
            total_vertical = alpha * total_vertical + (1 - alpha) * self.angle_history[-1][1]

        self.angle_history.append((total_horizontal, total_vertical))
        if len(self.angle_history) > 10:
            self.angle_history.pop(0)

        return (total_horizontal, total_vertical), gaze_vector

    def _dynamic_weight(self, angle, theta):
        """
        计算动态权重的饱和函数

        参数:
            angle: 头部姿态角度（yaw或pitch）
            theta: 对应的阈值参数

        返回:
            权重值，范围在[0.1, 0.5]之间
        """
        # 计算归一化角度值
        x = abs(angle) / theta

        # 饱和函数：σ(x) = min(0.5, max(0.1, x))
        weight = min(0.5, max(0.1, x))

        return weight

    def apply_roll_correction(self, gaze_data, roll_angle):
        """
        应用头部滚转校正到眼部特征点
        返回校正后的数据副本
        """
        if roll_angle is None or abs(roll_angle) <= self.params['roll_threshold']:
            return gaze_data

        # 创建数据副本避免修改原始数据
        corrected_data = copy.deepcopy(gaze_data)
        roll_rad = np.radians(roll_angle)
        cos_r = np.cos(roll_rad)
        sin_r = np.sin(roll_rad)

        # 校正双眼特征点
        for side in ["left", "right"]:
            corrected_data = self._correct_eye_features(
                corrected_data, side, roll_rad, cos_r, sin_r
            )

        # 重新计算视线比例
        corrected_data = self._recalculate_ratios(corrected_data)
        return corrected_data

    def _correct_eye_features(self, data, side, roll_rad, cos_r, sin_r):
        """校正单只眼睛的特征点"""
        prefix = f"{side}_"

        # --- 新增：检查关键坐标是否存在且非 None ---
        inner_x = data.get(f"{prefix}eye_inner_x")
        inner_y = data.get(f"{prefix}eye_inner_y")
        outer_x = data.get(f"{prefix}eye_outer_x")
        outer_y = data.get(f"{prefix}eye_outer_y")

        # 如果任何一个关键坐标是 None，则跳过该校正
        if any(coord is None for coord in [inner_x, inner_y, outer_x, outer_y]):
            print(f"[GAZE DIRECTION] Skipping {side} eye correction, missing eye corner coordinates.")
            # 返回原始数据，不进行任何修改
            return data
        # --- 新增结束 ---

        # 计算眼睛中心点（眼角中点）
        center_x = (inner_x + outer_x) / 2
        center_y = (inner_y + outer_y) / 2

        # --- 新增：检查瞳孔坐标 ---
        pupil_x = data.get(f"{prefix}pupil_x")
        pupil_y = data.get(f"{prefix}pupil_y")
        # 校正瞳孔位置 (仅当瞳孔坐标存在时)
        if pupil_x is not None and pupil_y is not None:
            px = pupil_x - center_x
            py = pupil_y - center_y
            data[f"{prefix}pupil_x"] = px * cos_r - py * sin_r + center_x
            data[f"{prefix}pupil_y"] = px * sin_r + py * cos_r + center_y
        # --- 新增结束 ---

        # --- 新增：检查眼睑位置 ---
        # 校正眼睑位置 (仅当眼睑坐标存在时)
        for lid_type in ["upper_lid", "lower_lid"]:
            lid_x_key = f"{prefix}eye_{lid_type}_x"
            lid_y_key = f"{prefix}eye_{lid_type}_y"
            lid_x = data.get(lid_x_key)
            lid_y = data.get(lid_y_key)

            if lid_x is not None and lid_y is not None:
                lx = lid_x - center_x
                ly = lid_y - center_y
                data[lid_x_key] = lx * cos_r - ly * sin_r + center_x
                data[lid_y_key] = lx * sin_r + ly * cos_r + center_y
        # --- 新增结束 ---

        return data

    def _recalculate_ratios(self, gaze_data):
        # 左眼水平比率：内眼角（左边界）→外眼角（右边界）
        left_h_ratio = self.calc_eye_ratio(
            gaze_data.get("left_pupil_x"),
            gaze_data.get("left_eye_inner_x"),  # 左边界（内眼角）
            gaze_data.get("left_eye_outer_x")  # 右边界（外眼角）
        )
        # 右眼水平比率：外眼角（左边界）→内眼角（右边界）
        right_h_ratio = self.calc_eye_ratio(
            gaze_data.get("right_pupil_x"),
            gaze_data.get("right_eye_outer_x"),  # 左边界（外眼角）
            gaze_data.get("right_eye_inner_x")  # 右边界（内眼角）
        )
        # 垂直比率：上眼睑（上边界）→下眼睑（下边界）
        left_v_ratio = self.calc_eye_ratio(
            gaze_data.get("left_pupil_y"),
            gaze_data.get("left_eye_upper_lid_y"),  # 上边界
            gaze_data.get("left_eye_lower_lid_y")  # 下边界
        )
        right_v_ratio = self.calc_eye_ratio(
            gaze_data.get("right_pupil_y"),
            gaze_data.get("right_eye_upper_lid_y"),
            gaze_data.get("right_eye_lower_lid_y")
        )

        # 新增：NaN值检测与替换，使用0.5作为默认值（避免注视点异常）
        def safe_ratio(ratio):
            if np.isnan(ratio) or ratio < 0.0 or ratio > 1.0:
                return 0.5
            return ratio

        left_h_ratio = safe_ratio(left_h_ratio)
        right_h_ratio = safe_ratio(right_h_ratio)
        left_v_ratio = safe_ratio(left_v_ratio)
        right_v_ratio = safe_ratio(right_v_ratio)

        # 计算平均值
        gaze_data["hr"] = (left_h_ratio + right_h_ratio) / 2
        gaze_data["vr"] = (left_v_ratio + right_v_ratio) / 2
        # 输出调试信息
        print("校正后的 hr:", gaze_data["hr"])
        print("校正后的 vr:", gaze_data["vr"])
        return gaze_data

    # 修改 gaze_direction.py 中的 calc_horizontal_angle 方法
    def calc_horizontal_angle(self, gaze_data):
        """计算水平视线角度"""
        # 分别计算左右眼的水平比率，使用安全访问方式
        left_h_ratio = self.calc_eye_ratio(
            gaze_data.get("left_pupil_x"),
            gaze_data.get("left_eye_inner_x"),
            gaze_data.get("left_eye_outer_x")
        )

        # right_h_ratio = self.calc_eye_ratio(
        #     gaze_data.get("right_pupil_x"),
        #     gaze_data.get("right_eye_inner_x"),
        #     gaze_data.get("right_eye_outer_x")
        # )
        # 修正右眼水平比率计算（左边界=外眼角，右边界=内眼角）
        right_h_ratio = self.calc_eye_ratio(
            gaze_data.get("right_pupil_x"),
            gaze_data.get("right_eye_outer_x"),  # 正确：右眼左边界=外眼角（45号点）
            gaze_data.get("right_eye_inner_x")  # 正确：右眼右边界=内眼角（42号点）
        )

        # 取平均值并转换为角度
        avg_ratio = (left_h_ratio + right_h_ratio) / 2

        # 应用用户自定义敏感度（如果在gaze_data中提供了）
        horizontal_sensitivity = self.params['horizontal_sensitivity']
        return horizontal_sensitivity * (avg_ratio - 0.5)


    def calc_vertical_angle(self, gaze_data):
        """重构：基于眼球中心计算垂直视线角度"""
        # 新增：打印调试信息，确认眼球中心Y坐标是否正确获取
        print(
            f"[DEBUG] left_eye_center_y: {gaze_data.get('left_eye_center_y')}, left_pupil_y: {gaze_data.get('left_pupil_y')}")
        print(
            f"[DEBUG] right_eye_center_y: {gaze_data.get('right_eye_center_y')}, right_pupil_y: {gaze_data.get('right_pupil_y')}")

        # 获取双眼眼球中心和瞳孔Y坐标
        left_eye_center_y = gaze_data.get("left_eye_center_y")
        left_pupil_y = gaze_data.get("left_pupil_y")
        right_eye_center_y = gaze_data.get("right_eye_center_y")
        right_pupil_y = gaze_data.get("right_pupil_y")

        # 双眼有效：取平均值（修复单眼依赖）
        if (left_eye_center_y is not None and left_pupil_y is not None and
                right_eye_center_y is not None and right_pupil_y is not None):
            # 左眼VR：(瞳孔Y - 眼球中心Y) / 眼睑间距（向上为负，向下为正）
            left_upper = gaze_data.get("left_eye_upper_lid_y")
            left_lower = gaze_data.get("left_eye_lower_lid_y")
            left_vr = (left_pupil_y - left_eye_center_y) / (left_lower - left_upper) if (
                                                                                                    left_lower - left_upper) != 0 else 0

            # 右眼VR
            right_upper = gaze_data.get("right_eye_upper_lid_y")
            right_lower = gaze_data.get("right_eye_lower_lid_y")
            right_vr = (right_pupil_y - right_eye_center_y) / (right_lower - right_upper) if (
                                                                                                         right_lower - right_upper) != 0 else 0

            vr = (left_vr + right_vr) / 2  # 双眼平均
        elif left_eye_center_y is not None and left_pupil_y is not None:
            # 仅左眼有效
            left_upper = gaze_data.get("left_eye_upper_lid_y")
            left_lower = gaze_data.get("left_eye_lower_lid_y")
            vr = (left_pupil_y - left_eye_center_y) / (left_lower - left_upper) if (left_lower - left_upper) != 0 else 0
        elif right_eye_center_y is not None and right_pupil_y is not None:
            # 仅右眼有效（兼容旧逻辑）
            right_upper = gaze_data.get("right_eye_upper_lid_y")
            right_lower = gaze_data.get("right_eye_lower_lid_y")
            vr = (right_pupil_y - right_eye_center_y) / (right_lower - right_upper) if (
                                                                                                   right_lower - right_upper) != 0 else 0
        else:
            vr = 0.0

        # 修正VR范围（-1~1，0为中心位置）
        # 不需要额外的范围限制，因为vr已经是相对值

        # 转换为角度（垂直敏感度，向上为正，向下为负）
        vertical_sensitivity = self.params['vertical_sensitivity']
        return -vertical_sensitivity * vr  # 符号修正：VR<0向上看角度为正

    def calc_eye_ratio(self, pupil_pos, inner_pos, outer_pos):
        """
        计算瞳孔在眼睛区域内的相对位置比率
        0.0 = 完全靠近内眼角/上眼睑
        1.0 = 完全靠近外眼角/下眼睑
        """
        if inner_pos is None or outer_pos is None or pupil_pos is None:
            return 0.5

        if outer_pos == inner_pos:
            return 0.5  # 避免除零错误

        return (pupil_pos - inner_pos) / (outer_pos - inner_pos)

    # def angles_to_vector(self, horizontal_deg, vertical_deg):
    #     """
    #     将水平和垂直角度转换为3D注视向量
    #
    #     参数:
    #         horizontal_deg: 水平角度（偏航角，-右 +左）
    #         vertical_deg: 垂直角度（俯仰角，-下 +上）
    #
    #     返回:
    #         np.array: 3D单位向量 [x, y, z]
    #     """
    #     # 转换为弧度
    #     yaw = np.radians(horizontal_deg)
    #     pitch = np.radians(vertical_deg)
    #
    #     # 计算3D向量分量
    #     x = -np.sin(yaw) * np.cos(pitch)  # 负号：屏幕坐标系中向右为正
    #     y = -np.sin(pitch)  # 负号：屏幕坐标系中向下为正
    #     z = np.cos(yaw) * np.cos(pitch)
    #
    #     # 归一化为单位向量
    #     norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    #     if norm > 0:
    #         return np.array([x / norm, y / norm, z / norm])
    #     return np.array([0, 0, 1])

    def angles_to_vector(self, horizontal_deg, vertical_deg):
        # 将水平和垂直角度转换为3D空间中的归一化方向向量，用于表示视线方向
        # 调整瞳孔移动的权重系数
        pupil_weight = 2.0

        #水平角度乘以瞳孔权重后转换为弧度，表示视线在水平方向上的偏移（yaw）
        yaw = np.radians(horizontal_deg * pupil_weight)
        pitch = np.radians(vertical_deg * pupil_weight)

        x = -np.sin(yaw) * np.cos(pitch)  # 视线方向在水平轴上的投影，向右为正
        y = -np.sin(pitch) # 视线方向在垂直轴上的投影，向下为正
        z = np.cos(yaw) * np.cos(pitch) # 视线方向在深度轴上的投影，正值表示向前

        # 归一化处理，计算向量的模长，确保长度为1
        norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        return np.array([x / norm, y / norm, z / norm]) if norm > 0 else np.array([0, 0, 1])

    def calculate_user_weight_thresholds(self, user_calib_samples):
        """
        根据用户校准样本计算个性化动态权重阈值

        参数:
            user_calib_samples: 用户校准阶段的样本数据列表
        """
        if not user_calib_samples or len(user_calib_samples) < 5:
            print("[WARN] 校准样本不足，使用默认阈值")
            return

        try:
            # 1. 统计用户校准阶段的头部运动幅度
            all_yaw = []
            all_pitch = []

            for sample in user_calib_samples:
                if isinstance(sample, dict):
                    # 处理字典格式样本
                    head_pose = sample.get('head_pose')
                    if head_pose and len(head_pose) >= 2:
                        all_pitch.append(head_pose[0])
                        all_yaw.append(head_pose[1])
                elif isinstance(sample, (list, tuple)) and len(sample) >= 3:
                    # 处理元组格式样本
                    head_pose = sample[2]  # 假设第三个元素是头部姿态
                    if head_pose and len(head_pose) >= 2:
                        all_pitch.append(head_pose[0])
                        all_yaw.append(head_pose[1])

            if len(all_yaw) < 3 or len(all_pitch) < 3:
                print("[WARN] 头部姿态数据不足，使用默认阈值")
                return

            # 2. 计算用户运动范围
            user_yaw_range = max(all_yaw) - min(all_yaw)
            user_pitch_range = max(all_pitch) - min(all_pitch)

            # 3. 设置个性化阈值（用户运动范围的1/2，确保覆盖80%日常运动）
            # 但设置合理的上下限，避免极端值
            self.params['theta_yaw'] = max(5.0, min(30.0, user_yaw_range / 2))
            self.params['theta_pitch'] = max(5.0, min(30.0, user_pitch_range / 2))

            print(f"[INFO] 个性化阈值设置完成:")
            print(f"       Theta_Yaw: {self.params['theta_yaw']:.2f} (基于用户Yaw范围: {user_yaw_range:.2f}度)")
            print(f"       Theta_Pitch: {self.params['theta_pitch']:.2f} (基于用户Pitch范围: {user_pitch_range:.2f}度)")

        except Exception as e:
            print(f"[ERROR] 计算个性化阈值失败: {e}")
            # 出错时使用默认值
            self.params['theta_yaw'] = 15.0
            self.params['theta_pitch'] = 15.0

    # 在 gaze_direction.py 的 GazeDirectionDetector 类中修改 compensate_head_yaw 方法
    def compensate_head_yaw(self, horizontal_ratio, head_yaw, target_x=None):
        """
        基于头部偏航角补偿水平注视比例

        参数:
            horizontal_ratio: 原始水平注视比例 (0-1)
            head_yaw: 头部偏航角（度，负值为左偏，正值为右偏）
            target_x: 目标点x坐标（可选，用于更精确补偿）

        返回:
            补偿后的水平注视比例
        """
        # 如果 horizontal_ratio 为 None，使用默认值 0.5（屏幕中心）
        if horizontal_ratio is None:
            horizontal_ratio = 0.5

        # 根据屏幕宽度调整补偿系数
        screen_w, _ = self._get_screen_resolution()

        # 动态补偿系数（每度偏航角影响的比例值）
        base_compensation_coeff = 0.008

        # 根据目标位置调整补偿系数
        if target_x is not None:
            # 如果目标在屏幕左侧，头部左偏时需要更强补偿
            # 如果目标在屏幕右侧，头部右偏时需要更强补偿
            target_position_factor = abs(target_x - screen_w / 2) / (screen_w / 2)
            compensation_coeff = base_compensation_coeff * (1 + target_position_factor * 0.5)
        else:
            compensation_coeff = base_compensation_coeff

        # 根据头部偏航角调整水平比例
        # 头部左偏（head_yaw负）时，实际注视点偏左，需要减小horizontal_ratio
        # 头部右偏（head_yaw正）时，实际注视点偏右，需要增大horizontal_ratio
        compensated_hr = horizontal_ratio - head_yaw * compensation_coeff

        # 限制在有效范围内
        compensated_hr = max(0.05, min(0.95, compensated_hr))

        return compensated_hr

    # 在 gaze_direction.py 的 GazeDirectionDetector 类中修改 calculate_screen_coords 方法
    def calculate_screen_coords(self, hr, vr, yaw, pitch, screen_w, screen_h):
        """
        根据HR/VR和头部姿态计算屏幕坐标

        参数:
            hr: 水平注视比例 (0-1)
            vr: 垂直注视比例 (0-1)
            yaw: 头部偏航角（度）
            pitch: 头部俯仰角（度）
            screen_w: 屏幕宽度
            screen_h: 屏幕高度

        返回:
            tuple: (screen_x, screen_y) 屏幕坐标
        """
        # 检查输入参数是否有效
        if hr is None or vr is None:
            # 如果 hr 或 vr 为 None，返回屏幕中心坐标
            return screen_w // 2, screen_h // 2

        # 修复后逻辑（正确）：先补偿hr再计算screen_x
        compensated_hr = self.compensate_head_yaw(hr, yaw)  # 应用Yaw角补偿
        screen_x = compensated_hr * screen_w  # 使用校正后的hr

        # 垂直方向保持现有逻辑（Pitch已验证有效）
        # 这里需要实现或调用垂直方向的补偿函数
        # 假设存在一个 compensate_head_pitch 方法
        compensated_vr = self._compensate_head_pitch(vr, pitch)  # 应用Pitch角补偿
        screen_y = compensated_vr * screen_h

        return screen_x, screen_y

    # 在 gaze_direction.py 的 GazeDirectionDetector 类中修改 _compensate_head_pitch 方法
    def _compensate_head_pitch(self, vertical_ratio, head_pitch):
        """
        基于头部俯仰角补偿垂直注视比例

        参数:
            vertical_ratio: 原始垂直注视比例 (0-1)
            head_pitch: 头部俯仰角（度，负值为下俯，正值为上仰）

        返回:
            补偿后的垂直注视比例
        """
        # 如果 vertical_ratio 为 None，使用默认值 0.5（屏幕中心）
        if vertical_ratio is None:
            vertical_ratio = 0.5

        # 动态补偿系数（每度俯仰角影响的比例值）
        base_compensation_coeff = 0.005

        # 根据头部俯仰角调整垂直比例
        # 头部上仰（head_pitch正）时，实际注视点偏上，需要减小vertical_ratio
        # 头部下俯（head_pitch负）时，实际注视点偏下，需要增大vertical_ratio
        compensated_vr = vertical_ratio - head_pitch * base_compensation_coeff

        # 限制在有效范围内
        compensated_vr = max(0.05, min(0.95, compensated_vr))

        return compensated_vr

    def _get_screen_resolution(self):
        """获取屏幕分辨率"""
        try:
            import ctypes
            user32 = ctypes.windll.user32
            return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        except:
            return 1920, 1080  # 默认分辨率


