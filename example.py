# example.py
# -*- coding: utf-8 -*-
import os
import sys

CALIBRATION_DIR = os.path.join(os.path.dirname(__file__), "calibration")
os.makedirs(CALIBRATION_DIR, exist_ok=True)
CALIBRATION_FILE = os.path.join(CALIBRATION_DIR, "calibration_data.npz")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import time
import ctypes
import numpy as np
from datetime import datetime

# 导入用户信息收集模块
from util.user_info import UserInfoCollector

from core.gaze_tracking import GazeTracking
from core.calibration_direction import DirectionCalibrator
from util.visualization import visualize_all, draw_calibration_points, resize_to_window, maximize_window, \
    generate_heatmap
from util.logger import GazeLogger
from core.gaze_direction import GazeDirectionDetector

import time

# 全局帧率控制
# TARGET_FPS = 15  # 降低目标帧率，使流程更用户友好
TARGET_FPS = 30
FRAME_TIME = 1.0 / TARGET_FPS  # 每帧时间


def control_frame_rate(start_time):
    """
    控制帧率的辅助函数
    在每个处理循环的末尾调用
    """
    process_time = time.time() - start_time
    if process_time < FRAME_TIME:
        time.sleep(FRAME_TIME - process_time)
    return time.time()  # 返回新的开始时间


# 获取屏幕分辨率
def get_screen_resolution():
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

# 加载校准文件
# CALIBRATION_FILE = "calibration_data.npz"

# CALIBRATION_FILE = get_user_calibration_file(user_info)

# 修改 save_calibration_data 函数
def save_calibration_data(calibrator, calibration_file_path):
    if calibrator.calibration_samples and len(calibrator.calibration_samples) > 0:
        try:
            # 确保目录存在
            calibration_dir = os.path.dirname(calibration_file_path)
            if calibration_dir:
                os.makedirs(calibration_dir, exist_ok=True)

            # 保存校准数据
            np.savez(
                calibration_file_path,
                calibration_samples=np.array(calibrator.calibration_samples, dtype=object),
                calibrated_ratio=calibrator.calibrated_ratio,
                calibration_ratios=np.array(calibrator.calibration_ratios, dtype=object) if hasattr(calibrator,
                                                                                                    'calibration_ratios') else np.array(
                    []),
                allow_pickle=True
            )
            print(f"[INFO] 校准数据已保存到 {calibration_file_path}")
            print(f"[DEBUG] 保存样本数量: {len(calibrator.calibration_samples)}")
        except Exception as e:
            print(f"[ERROR] 保存校准数据失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("[WARN] 无校准样本可保存")


# 修改 load_calibration_data 函数
def load_calibration_data(calibrator, calibration_file_path):
    """
    从文件加载校准数据

    参数:
        calibrator: DirectionCalibrator 校准器实例
        calibration_file_path: 校准文件路径

    返回:
        bool: 是否成功加载校准数据
    """
    if os.path.exists(calibration_file_path):
        try:
            # 允许pickle加载，因为保存时使用了allow_pickle=True
            data = np.load(calibration_file_path, allow_pickle=True)

            # 加载校准样本
            if "calibration_samples" in data:
                samples = data["calibration_samples"].tolist()
                if samples and len(samples) > 0:
                    calibrator.calibration_samples = samples
                    print(f"[INFO] 加载了 {len(samples)} 个校准样本")

            # 加载校准比例数据
            if "calibration_ratios" in data:
                ratios = data["calibration_ratios"].tolist()
                if ratios and len(ratios) > 0:
                    calibrator.calibration_ratios = ratios
                    print(f"[INFO] 加载了 {len(ratios)} 个比例校准数据")

            # 设置校准状态
            if "calibrated_ratio" in data:
                calibrator.calibrated_ratio = bool(data["calibrated_ratio"])
            else:
                # 如果没有明确的校准状态，根据数据判断
                calibrator.calibrated_ratio = (
                        hasattr(calibrator, 'calibration_ratios') and
                        len(calibrator.calibration_ratios) >= 9  # 至少需要9个点（3x3网格）
                )

            # 判断校准数据是否有效
            calibration_valid = False
            if hasattr(calibrator, 'calibration_ratios') and len(calibrator.calibration_ratios) >= 9:
                calibration_valid = True
            elif hasattr(calibrator, 'calibration_samples') and len(calibrator.calibration_samples) >= 9 * 30:
                calibration_valid = True

            # 根据数据内容设置最终的校准状态
            if calibration_valid:
                print(f"[INFO] 成功加载校准数据从 {calibration_file_path}")
                return True
            else:
                print(f"[WARN] 校准文件存在但未完成校准")
                # 即使数据不完整，我们也返回True表示文件存在并尝试加载
                # 但calibrated_ratio状态会反映实际校准情况
                return True

        except Exception as e:
            print(f"[ERROR] 加载校准数据失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[INFO] 校准文件 {calibration_file_path} 不存在")
    return False


def setup_console_logging(user_info, log_dir=None):
    """
    设置控制台日志重定向到用户文件夹

    参数:
        user_info: 用户信息字典
        log_dir: 指定日志目录，如果不提供则使用默认路径

    返回:
        log_file_path: 日志文件路径
    """
    # 如果提供了log_dir参数，则使用该目录，否则使用默认的user_logs目录
    if log_dir:
        user_log_dir = log_dir
    else:
        # 创建用户特定的日志目录
        if user_info and 'id' in user_info:
            user_id = user_info['id']
            user_log_dir = os.path.join("user_logs", f"user_{user_id}")
        else:
            user_log_dir = os.path.join("user_logs", "unknown_user")

    os.makedirs(user_log_dir, exist_ok=True)

    # 创建带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"console_log_{timestamp}.txt"
    log_file_path = os.path.join(user_log_dir, log_file_name)

    # 重定向stdout和stderr到文件
    class Logger:
        def __init__(self, stdout, filename):
            self.stdout = stdout
            self.logfile = open(filename, 'w', encoding='utf-8')

        def write(self, message):
            self.stdout.write(message)
            self.logfile.write(message)
            self.logfile.flush()

        def flush(self):
            self.stdout.flush()
            self.logfile.flush()

        def close(self):
            self.logfile.close()

    # 应用重定向
    sys.stdout = Logger(sys.stdout, log_file_path)
    sys.stderr = Logger(sys.stderr, log_file_path)

    print(f"[INFO] 控制台日志将保存到: {log_file_path}")
    return log_file_path



# --------------------------初始化阶段---------------------------
# ======【主程序逻辑】======
# 首先收集用户信息
print("[INFO] 启动用户信息收集...")
user_collector = UserInfoCollector()
user_info = user_collector.collect_user_info()

# 获取屏幕分辨率
screen_w, screen_h = get_screen_resolution()

# 在获取用户信息后设置主视眼
if user_info and 'dominant_eye' in user_info:
    primary_eye = user_info['dominant_eye']
    direction_calibrator = DirectionCalibrator(primary_eye=primary_eye)
    print(f"[INFO] 使用{primary_eye}眼作为主视眼")
else:
    direction_calibrator = DirectionCalibrator(primary_eye='right')
    print("[INFO] 使用默认右眼作为主视眼")

if user_info is None:
    print("[INFO] 用户取消操作，程序退出")
    exit(0)


# 初始化日志器
logger = GazeLogger(output_dir="dataset", user_info=user_info)

# 将控制台日志保存到与用户数据相同的目录
log_file_path = setup_console_logging(user_info, logger.log_dir)


print(f"[INFO] 用户信息收集完成: {user_info}")

# 获取屏幕分辨率
screen_w, screen_h = get_screen_resolution()

# 启动摄像头和眼动追踪器
gaze = GazeTracking(direction_calibrator=direction_calibrator)
webcam = cv2.VideoCapture(0)

# 设置摄像头帧率
webcam.set(cv2.CAP_PROP_FPS, TARGET_FPS)
# webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# 初始化日志器
logger = GazeLogger(output_dir="dataset", user_info=user_info)


# 修改校准文件路径，使其与用户关联
def get_user_calibration_file(user_info):
    """
    为特定用户生成校准文件路径
    :param user_info: 用户信息字典
    :return: 校准文件路径
    """
    if user_info and 'id' in user_info:
        user_id = user_info['id']
        # 创建用户特定的校准目录
        user_calibration_dir = os.path.join(CALIBRATION_DIR, f"user_{user_id}")
        os.makedirs(user_calibration_dir, exist_ok=True)
        return os.path.join(user_calibration_dir, "calibration_data.npz")
    else:
        return os.path.join(CALIBRATION_DIR, "calibration_data.npz")


# 为当前用户设置校准文件路径
USER_CALIBRATION_FILE = get_user_calibration_file(user_info)


# ------------------------交互式校准-----------------------
def interactive_calibration_mode(gaze, webcam, logger, direction_calibrator):
    """
    交互式手动校准模式
    """
    print("[INFO] 进入交互式校准模式...")

    # 获取屏幕分辨率
    screen_w, screen_h = get_screen_resolution()

    # 创建校准窗口
    calibration_window_name = "Interactive Calibration"
    cv2.namedWindow(calibration_window_name, cv2.WINDOW_NORMAL)
    maximize_window(calibration_window_name)

    # 定义校准点
    calibration_points = [
        (100, 100), (screen_w // 2, 100), (screen_w - 100, 100),
        (100, screen_h // 2), (screen_w // 2, screen_h // 2), (screen_w - 100, screen_h // 2),
        (100, screen_h - 100), (screen_w // 2, screen_h - 100), (screen_w - 100, screen_h - 100)
    ]

    # 添加额外的校准点以提高精度
    additional_points = [
        (screen_w // 4, screen_h // 4), (3 * screen_w // 4, screen_h // 4),
        (screen_w // 4, 3 * screen_h // 4), (3 * screen_w // 4, 3 * screen_h // 4)
    ]

    all_points = calibration_points + additional_points
    collected_samples = []

    print(f"[INFO] 将显示 {len(all_points)} 个校准点，请注视每个点并按空格键确认")
    print("[INFO] 每个点需要稳定注视2秒")

    for i, (target_x, target_y) in enumerate(all_points):
        print(f"[INFO] 请注视点 {i + 1}/{len(all_points)}: ({target_x}, {target_y})")

        # 显示校准点
        frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        cv2.circle(frame, (target_x, target_y), 20, (0, 255, 0), -1)
        cv2.circle(frame, (target_x, target_y), 25, (255, 255, 255), 2)
        cv2.putText(frame, f"Point {i + 1}/{len(all_points)}",
                    (target_x + 30, target_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Press SPACE to confirm, ESC to exit",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(calibration_window_name, frame)

        # 给用户时间准备
        cv2.waitKey(1000)

        # 等待用户确认
        confirmed = False
        sample_count = 0
        required_samples = 20  # 每个点采集20个样本
        start_collection_time = time.time()

        while not confirmed:
            frame_start_time = time.time()
            ret, raw_frame = webcam.read()
            if not ret:
                frame_start_time = control_frame_rate(frame_start_time)
                continue

            gaze.refresh(raw_frame)
            gaze_data = gaze.process_frame()

            # 显示摄像头画面
            annotated_frame = visualize_all(raw_frame, gaze_data, mode="tracking")
            resized_frame = resize_to_window(annotated_frame, calibration_window_name, (screen_w, screen_h))
            # cv2.imshow("Camera View", resized_frame)

            # 显示倒计时
            elapsed = time.time() - start_collection_time
            remaining = max(0, 2 - elapsed)  # 2秒采集时间
            countdown_frame = frame.copy()
            cv2.putText(countdown_frame, f"Collecting: {remaining:.1f}s",
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow(calibration_window_name, countdown_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # 空格键确认
                confirmed = True
                break
            elif key == 27:  # ESC键退出
                cv2.destroyWindow(calibration_window_name)
                cv2.destroyWindow("Camera View")
                return False

            # 自动采集样本（无需按键）
            if sample_count < required_samples:
                if gaze_data["hr"] is not None and gaze_data["vr"] is not None:
                    sample = {
                        'target_point': (target_x / screen_w, target_y / screen_h),
                        'hr_vr': (gaze_data['hr'], gaze_data['vr']),
                        'head_pose': (gaze_data['head_pitch'],
                                      gaze_data['head_yaw'],
                                      gaze_data['head_roll']),
                        'gaze_vector': (gaze_data['gaze_vx'],
                                        gaze_data['gaze_vy'],
                                        gaze_data['gaze_vz']),
                        'pupil_coords': {
                            'left': gaze_data['left_pupil'],
                            'right': gaze_data['right_pupil']
                        }
                    }
                    collected_samples.append(sample)
                    sample_count += 1
                    print(f"[DEBUG] 收集样本 {sample_count}/{required_samples} for point {i + 1}")

            # 控制帧率
            frame_start_time = control_frame_rate(frame_start_time)

            # 如果采集时间超过2秒，自动继续
            if time.time() - start_collection_time >= 2.0:
                confirmed = True
                break

    # 使用收集到的数据改进校准
    if len(collected_samples) >= 15:  # 确保有足够的样本
        improve_calibration_with_samples(direction_calibrator, collected_samples)
        try:
            direction_calibrator.train_advanced_mapping_model()
        except AttributeError:
            print("[WARN] 高级映射模型训练方法未找到")
    else:
        print("[WARN] 样本数量不足，跳过高级模型训练")

    cv2.destroyWindow(calibration_window_name)
    cv2.destroyWindow("Camera View")
    return True


# 修改 example.py 中的 improve_calibration_with_samples 函数
def improve_calibration_with_samples(direction_calibrator, samples):
    """
    使用收集到的样本改进校准
    """
    print(f"[INFO] 使用 {len(samples)} 个样本改进校准...")

    # 清除旧的校准数据
    direction_calibrator.calibration_ratios = []

    # 添加新样本（减少重复检查的严格性）
    added_samples = 0
    for sample in samples:
        target_point = sample['target_point']
        hr_vr = sample['hr_vr']

        # 新增：检查hr_vr是否有效
        if hr_vr is None or not isinstance(hr_vr, (list, tuple)) or len(hr_vr) < 2:
            print(f"[WARN] 无效的hr_vr数据: {hr_vr}，跳过该样本")
            continue

        if any(v is None for v in hr_vr[:2]):
            print(f"[WARN] hr_vr包含None值: {hr_vr}，跳过该样本")
            continue

        # 简化重复检查 - 统一处理不同格式
        should_add = True
        if len(direction_calibrator.calibration_ratios) > 0:
            try:
                # 统一处理不同格式的样本
                existing_ratios = []
                for existing_sample in direction_calibrator.calibration_ratios:
                    if isinstance(existing_sample, dict):
                        # 新格式：字典形式
                        eye_ratio = existing_sample.get('eye_ratio')
                    elif isinstance(existing_sample, (list, tuple)) and len(existing_sample) >= 2:
                        # 旧格式：列表形式
                        eye_ratio = existing_sample[1]
                    else:
                        continue

                    if eye_ratio and len(eye_ratio) >= 2:
                        existing_ratios.append(eye_ratio)

                if existing_ratios:
                    existing_ratios = np.array(existing_ratios)
                    distances = np.sqrt(np.sum((existing_ratios - np.array([hr_vr[0], hr_vr[1]])) ** 2, axis=1))
                    min_distance = np.min(distances) if len(distances) > 0 else float('inf')
                    # 放宽重复检查阈值
                    if min_distance < 0.01:
                        should_add = False
                        print(f"[DEBUG] 样本过于接近已有样本，跳过添加")
            except (KeyError, IndexError, TypeError, ValueError) as e:
                print(f"[WARN] 检查样本重复时出错: {e}，重置校准数据")
                direction_calibrator.calibration_ratios = []
                should_add = True

        if should_add:
            direction_calibrator.add_calibration_sample(target_point, hr_vr)
            added_samples += 1
            print(f"[DEBUG] 添加样本 {added_samples}: target={target_point}, ratio={hr_vr}")

    print(f"[INFO] 实际添加了 {added_samples} 个样本")

    # 重新训练模型
    if added_samples > 0:
        try:
            direction_calibrator.finalize_ratio_calibration()
            print("[INFO] 校准模型重新训练完成")
        except Exception as e:
            print(f"[ERROR] 重新训练校准模型失败: {e}")
    else:
        print("[WARN] 没有有效样本添加，跳过模型训练")

    # 训练头部姿态区域模型
    print("[INFO] 训练头部姿态区域模型...")
    try:
        direction_calibrator.train_head_pose_zone_models()
        print("[INFO] 头部姿态区域模型训练完成")
    except Exception as e:
        print(f"[ERROR] 训练头部姿态区域模型失败: {e}")

    # 保存头部姿态区域模型
    try:
        import pickle
        model_data = {
            'head_pose_zone_models': direction_calibrator.head_pose_zone_models,
            'head_pose_zones': direction_calibrator.head_pose_zones,
            'head_pose_zone_samples': direction_calibrator.head_pose_zone_samples
        }
        model_file_path = USER_CALIBRATION_FILE.replace('.npz', '_head_pose_models.pkl')
        with open(model_file_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"[INFO] 头部姿态区域模型已保存到 {model_file_path}")
    except Exception as e:
        print(f"[ERROR] 保存头部姿态区域模型失败: {e}")

    print("[INFO] 校准改进完成")


# --------------------------交互式校准阶段--------------------------
print("[INFO] 检查是否需要进行交互式校准...")
user_choice = input("是否进行交互式校准以提高精度？(y/n): ").strip().lower()
if user_choice == 'y':
    print("[INFO] 启动交互式校准...")
    success = interactive_calibration_mode(gaze, webcam, logger, direction_calibrator)
    if success:
        print("[INFO] 交互式校准完成")
        # 保存改进后的校准数据
        save_calibration_data(direction_calibrator, USER_CALIBRATION_FILE)
    else:
        print("[INFO] 交互式校准被取消")

# --------------------------校准阶段--------------------------
# 尝试加载已有校准数据
calibration_loaded = load_calibration_data(direction_calibrator, USER_CALIBRATION_FILE)

# 检查校准数据是否有效
calibration_valid = False
if calibration_loaded:
    if hasattr(direction_calibrator, 'calibration_ratios') and len(direction_calibrator.calibration_ratios) >= 9:
        calibration_valid = True
    elif hasattr(direction_calibrator, 'calibration_samples') and len(
            direction_calibrator.calibration_samples) >= 9 * 30:
        calibration_valid = True

skip_calibration = False
if calibration_loaded and calibration_valid:
    print(f"[INFO] Using cached calibration data for user {user_info.get('id', 'unknown')}.")
    print("[INFO] Calibration already completed. Skipping calibration phase.")
    skip_calibration = True
else:
    print("[INFO] Starting full-screen calibration...")

# 只有在需要校准时才执行校准过程
if not skip_calibration:
    calibration_window_name = "Calibration"
    cv2.namedWindow(calibration_window_name, cv2.WINDOW_NORMAL)
    maximize_window(calibration_window_name)

    # 初始化视频录制
    _, first_frame = webcam.read()
    if first_frame is not None:
        dummy_screen_frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        actual_fps = int(webcam.get(cv2.CAP_PROP_FPS))
        if actual_fps <= 0:
            actual_fps = TARGET_FPS
        logger.init_video_writers(dummy_screen_frame.shape, first_frame.shape, fps=actual_fps)

    logger.set_current_stage('calibration')

    # 找到校准阶段的参数设置
    total_points = 9
    samples_per_point = 50  # 保持合理的采样数量
    min_valid_samples = 30  # 最小有效样本数
    point_index = 0
    sample_count = 0
    valid_samples_count = 0

    # 校准循环
    while point_index < total_points:
        frame_start_time = time.time()
        frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        coords = draw_calibration_points(frame, point_index)
        target_x, target_y = coords[point_index]
        direction_calibrator.current_target = (target_x / screen_w, target_y / screen_h)

        # 显示当前点信息
        cv2.putText(frame, f"Point {point_index + 1}/{total_points}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Please focus on the point", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(calibration_window_name, frame)

        # 给用户时间准备
        cv2.waitKey(1000)

        # 开始采集样本
        # # 修改校准点采集循环中的时间控制逻辑
        # start_collection_time = time.time()
        # point_duration = 6.0  # 固定每个点注视3秒
        # 修改校准点采集循环中的时间控制逻辑
        start_collection_time = time.time()
        # 对屏幕四角等关键区域延长采集时间
        if point_index in [0, 2, 6, 8]:  # 四角点（根据实际点的索引调整）
            point_duration = 9.0  # 四角点注视时间延长至4.5秒
        else:
            point_duration = 6.0  # 其他点保持3秒

        while (sample_count < samples_per_point and valid_samples_count < min_valid_samples) and (
                time.time() - start_collection_time < point_duration):
            frame_start_time = time.time()
            _, raw_frame = webcam.read()
            gaze.refresh(raw_frame)

            # 添加这三行代码来传递真实目标坐标
            gaze.gaze_data = {
                "target_coords": (target_x, target_y),  # 传递真实目标坐标
                "direction": f"calibration_point_{point_index + 1}"
            }
            gaze_data = gaze.process_frame()

            # 计算屏幕坐标和误差
            screen_coords = gaze_data.get("screen_coords")
            error_x = abs(screen_coords[0] - target_x) if screen_coords else ''
            error_y = abs(screen_coords[1] - target_y) if screen_coords else ''

            # 校准阶段循环内，在构建log_data前添加：
            # 1. 计算HR/VR
            hr = gaze.horizontal_ratio()  # 调用GazeTracking的HR计算方法
            vr = gaze.vertical_ratio()  # 调用GazeTracking的VR计算方法
            # 2. 获取瞳孔坐标
            left_pupil = gaze.pupil_left_coords()  # 左眼瞳孔坐标
            right_pupil = gaze.pupil_right_coords()  # 右眼瞳孔坐标

            log_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                # "hr": gaze_data["hr"],
                # "vr": gaze_data["vr"],
                "hr": hr if hr is not None else "",
                "vr": vr if vr is not None else "",
                "screen_x": screen_coords[0] if screen_coords else '',
                "screen_y": screen_coords[1] if screen_coords else '',
                "target_x": target_x,
                "target_y": target_y,
                "error_x": error_x,
                "error_y": error_y,
                "head_pitch": gaze_data["head_pitch"],
                "head_yaw": gaze_data["head_yaw"],
                "head_roll": gaze_data["head_roll"],
                "gaze_vx": gaze_data["gaze_vx"],
                "gaze_vy": gaze_data["gaze_vy"],
                "gaze_vz": gaze_data["gaze_vz"],
                "left_pupil_x": left_pupil[0] if left_pupil else "",
                "left_pupil_y": left_pupil[1] if left_pupil else "",
                "right_pupil_x": right_pupil[0] if right_pupil else "",
                "right_pupil_y": right_pupil[1] if right_pupil else "",
                "left_eye_inner_x": gaze_data["left_eye_inner_x"],
                "left_eye_inner_y": gaze_data["left_eye_inner_y"],
                "left_eye_outer_x": gaze_data["left_eye_outer_x"],
                "left_eye_outer_y": gaze_data["left_eye_outer_y"],
                "right_eye_inner_x": gaze_data["right_eye_inner_x"],
                "right_eye_inner_y": gaze_data["right_eye_inner_y"],
                "right_eye_outer_x": gaze_data["right_eye_outer_x"],
                "right_eye_outer_y": gaze_data["right_eye_outer_y"],
                "left_eye_upper_lid_x": gaze_data["left_eye_upper_lid_x"],
                "left_eye_upper_lid_y": gaze_data["left_eye_upper_lid_y"],
                "left_eye_lower_lid_x": gaze_data["left_eye_lower_lid_x"],
                "left_eye_lower_lid_y": gaze_data["left_eye_lower_lid_y"],
                "right_eye_upper_lid_x": gaze_data["right_eye_upper_lid_x"],
                "right_eye_upper_lid_y": gaze_data["right_eye_upper_lid_y"],
                "right_eye_lower_lid_x": gaze_data["right_eye_lower_lid_x"],
                "right_eye_lower_lid_y": gaze_data["right_eye_lower_lid_y"],
                "is_blinking": gaze_data["is_blinking"],
                "direction": f"calibration_point_{point_index + 1}",
                "screen_coords": screen_coords,
                "target_coords": (target_x, target_y)
            }

            logger.log(log_data)
            logger.auto_flush()

            # 写入视频帧
            screen_frame_for_video = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            draw_calibration_points(screen_frame_for_video, point_index)
            logger.write_video_frame(screen_frame=screen_frame_for_video, user_frame=raw_frame)

            # 数据质量检查 - 添加头部姿态信息
            if isinstance(gaze_data["hr"], float) and isinstance(gaze_data["vr"], float):
                # 放宽范围但排除极端值（0.01~0.99，避免HR=0.0的无效样本）
                if 0.01 <= gaze_data["hr"] <= 0.99 and 0.01 <= gaze_data["vr"] <= 0.99:
                    valid_samples_count += 1
                    hr = gaze_data["hr"]
                    vr = gaze_data["vr"]
                    target_point = (target_x / screen_w, target_y / screen_h)
                    eye_ratio = (hr, vr)

                    # 添加头部姿态信息到校准样本
                    head_pose = (
                        gaze_data.get("head_pitch", 0),
                        gaze_data.get("head_yaw", 0),
                        gaze_data.get("head_roll", 0)
                    )

                    # 修改调用方式，传递头部姿态信息
                    direction_calibrator.add_calibration_sample(
                        target_point,
                        eye_ratio,
                        head_pose=head_pose
                    )

                    debug_entry = direction_calibrator.create_debug_entry(
                        gaze_data, log_data,
                        target_x, target_y,
                        screen_w, screen_h,
                        screen_coords
                    )
                    direction_calibrator.add_sample_with_debug(hr, vr, debug_entry)
                else:
                    print(f"[WARN] Invalid ratio: hr={gaze_data['hr']:.2f}, vr={gaze_data['vr']:.2f}")
                    # 打印无效样本原因，便于调试
                    print(f"[WARN] 跳过无效样本: HR={gaze_data['hr']:.2f}, VR={gaze_data['vr']:.2f}（超出0.01~0.99范围）")
                    logger.log_error(f"Invalid ratio - hr:{gaze_data['hr']:.2f}, vr:{gaze_data['vr']:.2f}")

            # 新增：打印实时进度，便于调试
            if sample_count % 5 == 0:
                print(
                    f"[DEBUG] 第{point_index + 1}个点进度: 总样本={sample_count}/{samples_per_point}, 有效样本={valid_samples_count}/{min_valid_samples}, 已耗时={time.time() - start_collection_time:.1f}s")

            # 可视化
            annotated_frame = visualize_all(
                raw_frame,
                gaze_data,
                mode="calibration",
                calibration_index=point_index
            )

            # 显示采集进度和头部运动提示
            progress_frame = frame.copy()
            elapsed = time.time() - start_collection_time
            cv2.putText(progress_frame, f"Collecting: {sample_count + 1}/{samples_per_point}",
                        (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(progress_frame, f"Elapsed: {elapsed:.1f}s",
                        (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 添加头部运动提示
            # if sample_count % 15 == 0 and sample_count != 0:
            #     cv2.putText(progress_frame, "轻微移动头部（左右/上下）",
            #                 (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # # 修改为（每10帧提示）：
            # if sample_count % 10 == 0 and sample_count != 0:
            #     cv2.putText(progr-ess_frame, "Please move your head slightly",
            #                 (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            #
            #
            # cv2.imshow(calibration_window_name, progress_frame)

            # 校准循环内：每采集10个样本，触发一次头部运动提示（延长显示+英文指引，避免乱码）
            if sample_count % 10 == 0 and sample_count != 0:
                # 1. 分步骤的头部运动英文提示（清晰指引，强调“保持注视红色点”）
                # 步骤设计：正视→左转→恢复→右转→恢复→抬头→恢复→低头，覆盖全姿态
                head_movement_steps = [
                    "Step 1: Look directly at the red point (hold for 3s)",
                    "Step 2: Turn head LEFT as far as possible (keep looking at red point, 3s)",
                    "Step 3: Return to looking directly at red point (hold for 2s)",
                    "Step 4: Turn head RIGHT as far as possible (keep looking at red point, 3s)",
                    "Step 5: Return to looking directly at red point (hold for 2s)",
                    "Step 6: Tilt head UP slowly (keep looking at red point, 3s)",
                    "Step 7: Return to looking directly at red point (hold for 2s)",
                    "Step 8: Tilt head DOWN slowly (keep looking at red point, 3s)"
                ]

                # 2. 逐个步骤显示提示，每个步骤显示足够时长（3-5秒，用户有时间反应）
                for step_text in head_movement_steps:
                    # 创建提示画面：浅灰色背景（不刺眼）+ 红色校准点（明确注视目标）
                    instruction_frame = np.ones((screen_h, screen_w, 3), dtype=np.uint8) * 240  # Light gray background
                    # 绘制红色校准点（突出，确保用户能找到注视目标）
                    cv2.circle(instruction_frame, (target_x, target_y), 20, (0, 0, 255), -1)  # Solid red point
                    cv2.circle(instruction_frame, (target_x, target_y), 25, (255, 255, 255), 3)  # White border

                    # 绘制英文提示文字（屏幕中间偏左，大字体+黑色加粗，清晰易读）
                    cv2.putText(
                        instruction_frame, step_text,
                        (screen_w // 6, screen_h // 2),  # Position: avoid covering the red point
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2,  # Font scale: larger than before
                        (0, 0, 0), 3  # Black color + bold (thickness=3)
                    )
                    # 底部额外提醒：强化“保持注视红点”的核心要求
                    cv2.putText(
                        instruction_frame, "IMPORTANT: Keep your eyes on the RED point during all steps!",
                        (screen_w // 8, screen_h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 0, 0), 2  # Blue color: highlight key reminder
                    )

                    # 3. 显示当前步骤提示，按步骤设置时长（关键动作3秒，恢复动作2秒）
                    # Determine display time: 3s for key moves, 2s for return moves
                    display_time = 3000 if any(
                        keyword in step_text for keyword in ["LEFT", "RIGHT", "UP", "DOWN"]) else 2000
                    cv2.imshow(calibration_window_name, instruction_frame)
                    cv2.waitKey(display_time)  # Show for 2-3 seconds (enough to read and react)

                    # 4. 过渡画面：恢复显示校准点，避免用户迷失目标
                    transition_frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                    cv2.circle(transition_frame, (target_x, target_y), 20, (0, 255, 0),
                               -1)  # Green point for transition
                    cv2.putText(transition_frame, "Continue focusing...", (50, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imshow(calibration_window_name, transition_frame)
                    cv2.waitKey(1000)  # Short transition (1s) to help user reorient

            # 使用自适应缩放函数
            resized_frame = resize_to_window(annotated_frame, calibration_window_name, (screen_w, screen_h))
            # cv2.imshow("Camera View", resized_frame)

            sample_count += 1

            # 控制帧率
            frame_start_time = control_frame_rate(frame_start_time)

            key = cv2.waitKey(1)
            if key == 27:
                break

        # 移动到下一个点
        sample_count = 0
        valid_samples_count = 0
        point_index += 1

        key = cv2.waitKey(1)
        if key == 27:
            break

    # 完成校准处理
    direction_calibrator.finalize_ratio_calibration()


    # def evaluate_calibration_quality(calibrator):
    #     """评估校准质量并提供改进建议"""
    #     if len(calibrator.calibration_ratios) < 9:
    #         return False, "校准点数量不足"
    #
    #     # 检查校准点分布
    #     targets = np.array([pair[0] for pair in calibrator.calibration_ratios])
    #     x_std = np.std(targets[:, 0])
    #     y_std = np.std(targets[:, 1])
    #
    #     if x_std < 0.2 or y_std < 0.2:
    #         return False, "校准点分布不均匀"
    #
    #     # 检查样本一致性（标准差）
    #     ratios = np.array([pair[1] for pair in calibrator.calibration_ratios])
    #     hr_std = np.std(ratios[:, 0])
    #     vr_std = np.std(ratios[:, 1])
    #
    #     if hr_std < 0.05 or vr_std < 0.05:
    #         return False, "样本变化范围过小"
    #
    #     return True, "校准质量良好"


    def evaluate_calibration_quality(calibrator):
        """评估校准质量并提供改进建议"""
        if len(calibrator.calibration_ratios) < 9:
            return False, "校准点数量不足"

        # 检查校准点分布
        targets = []
        for sample in calibrator.calibration_ratios:
            if isinstance(sample, dict):
                # 新格式：字典形式
                targets.append(sample.get('target_point'))
            elif isinstance(sample, (list, tuple)) and len(sample) >= 1:
                # 旧格式：列表形式
                targets.append(sample[0])

        if len(targets) < 9:
            return False, "有效校准点数量不足"

        targets = np.array(targets)
        x_std = np.std(targets[:, 0])
        y_std = np.std(targets[:, 1])

        if x_std < 0.2 or y_std < 0.2:
            return False, "校准点分布不均匀"

        # 检查样本一致性（标准差）
        ratios = []
        for sample in calibrator.calibration_ratios:
            if isinstance(sample, dict):
                # 新格式：字典形式
                ratios.append(sample.get('eye_ratio'))
            elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
                # 旧格式：列表形式
                ratios.append(sample[1])

        if len(ratios) < 9:
            return False, "有效样本数量不足"

        ratios = np.array(ratios)
        hr_std = np.std(ratios[:, 0])
        vr_std = np.std(ratios[:, 1])

        if hr_std < 0.05 or vr_std < 0.05:
            return False, "样本变化范围过小"

        return True, "校准质量良好"


    # 评估并保存校准数据
    is_good, message = evaluate_calibration_quality(direction_calibrator)
    print(f"[CALIBRATION] {message}")

    if is_good:
        save_calibration_data(direction_calibrator, USER_CALIBRATION_FILE)
        print("✅ 比例校准已完成")
    else:
        print("[CALIBRATION] 建议重新校准以提高精度")
        save_calibration_data(direction_calibrator, USER_CALIBRATION_FILE)

    # 关闭校准窗口
    cv2.destroyWindow(calibration_window_name)
    cv2.destroyWindow("Camera View")

    # 校准完成后短暂暂停
    time.sleep(1)

    # 在校准阶段完成后添加
    print("[INFO] 正在提取校准阶段关键日志...")
    key_log_path = logger.extract_key_logs_on_exit()

    # 在校准完成后的代码段中添加（在校准阶段完成后）
    def apply_personalized_thresholds(direction_calibrator, calibration_samples):
        """
        应用个性化阈值到注视方向检测器

        参数:
            direction_calibrator: 方向校准器实例
            calibration_samples: 校准样本数据
        """
        try:
            # 获取注视方向检测器实例
            if hasattr(direction_calibrator, 'gaze_direction_detector'):
                detector = direction_calibrator.gaze_direction_detector
            else:
                # 如果没有直接引用，创建一个新的实例用于计算
                from core.gaze_direction import GazeDirectionDetector
                detector = GazeDirectionDetector()

            # 计算并应用个性化阈值
            detector.calculate_user_weight_thresholds(calibration_samples)

            # 将计算得到的阈值应用到方向校准器
            if hasattr(direction_calibrator, 'gaze_direction_detector'):
                direction_calibrator.gaze_direction_detector.params['theta_yaw'] = detector.params['theta_yaw']
                direction_calibrator.gaze_direction_detector.params['theta_pitch'] = detector.params['theta_pitch']

            print(
                f"[INFO] 个性化动态权重阈值已应用: θ_yaw={detector.params['theta_yaw']:.2f}, θ_pitch={detector.params['theta_pitch']:.2f}")

        except Exception as e:
            print(f"[ERROR] 应用个性化阈值失败: {e}")


    # 在校准阶段完成后调用
    if hasattr(direction_calibrator, 'calibration_samples') and len(direction_calibrator.calibration_samples) > 0:
        apply_personalized_thresholds(direction_calibrator, direction_calibrator.calibration_samples)

# --------------------------验证阶段--------------------------
print("[INFO] Starting validation phase...")
# try:
#     logger.set_current_stage('validation')
#
#     from util.validation import run_validation
#
#     validation_results = run_validation(gaze, webcam, logger, direction_calibrator)
#     print(f"[INFO] Validation completed with {len(validation_results)} points")
# except Exception as e:
#     print(f"[ERROR] Validation failed: {e}")
#     import traceback
#
#     traceback.print_exc()
#
# # 验证完成后短暂暂停
# time.sleep(1)
try:
    logger.set_current_stage('validation')

    from util.validation import run_validation

    # 添加重试机制处理NaN错误
    validation_attempts = 3  # 最多重试3次
    for attempt in range(validation_attempts):
        try:
            validation_results = run_validation(gaze, webcam, logger, direction_calibrator)
            print(f"[INFO] Validation completed with {len(validation_results)} points")
            break  # 成功则退出重试循环
        except ValueError as e:
            if attempt < validation_attempts - 1:
                print(f"[WARNING] Validation attempt {attempt + 1} failed (NaN error), retrying...")
                time.sleep(1)  # 重试前等待1秒
            else:
                raise e  # 最后一次尝试失败则抛出异常

    # 在验证阶段正常完成后提取关键日志
    print("[INFO] 正在提取验证阶段关键日志...")
    key_log_path = logger.extract_key_logs_on_exit()

except Exception as e:
    print(f"[ERROR] Validation failed: {e}")
    import traceback

    traceback.print_exc()
    # 可选：强制重新校准
    print("[INFO] Retrying calibration due to validation failure...")
    interactive_calibration_mode(gaze, webcam, logger, direction_calibrator)

# ------------------------实时追踪阶段--------------------------
print("[INFO] Starting real-time gaze tracking...")
tracking_window_name = 'Gaze Tracking'
cv2.namedWindow(tracking_window_name, cv2.WINDOW_NORMAL)
maximize_window(tracking_window_name)

logger.set_current_stage('tracking')

if 'gaze_detector' not in globals():
    gaze_detector = GazeDirectionDetector()  # 添加这行作为保险

# ======【追踪循环】======
while True:
    frame_start_time = time.time()
    ret, raw_frame = webcam.read()
    if not ret:
        frame_start_time = control_frame_rate(frame_start_time)
        break

    # 处理帧数据
    gaze.refresh(raw_frame)
    gaze_data = gaze.process_frame()

    # 添加动态校准提示
    if hasattr(gaze, 'dynamic_calibration_triggered') and gaze.dynamic_calibration_triggered:
        # 在画面上显示提示信息
        if hasattr(gaze, 'dynamic_calibration_point') and gaze.dynamic_calibration_point:
            # 绘制校准点
            cv2.circle(raw_frame, gaze.dynamic_calibration_point, 20, (0, 0, 255), -1)
            cv2.circle(raw_frame, gaze.dynamic_calibration_point, 25, (255, 255, 255), 2)

            # 显示提示文本
            cv2.putText(raw_frame, "Dynamic Calibration - Please focus on the red point",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(raw_frame, "Collecting samples for auto-calibration...",
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 使用注视方向检测器
    gaze_angles, gaze_vector = gaze_detector.detect_gaze_direction(gaze_data)

    # 将3D注视向量添加到gaze_data中
    gaze_data["gaze_vector"] = gaze_vector

    # 获取实时注视点坐标
    screen_coords = gaze_data.get("screen_coords")

    # 获取目标点坐标（如果有的话）
    target_coords = gaze_data.get("target_coords")

    # 计算误差
    error_x = abs(screen_coords[0] - target_coords[0]) if screen_coords and target_coords else ''
    error_y = abs(screen_coords[1] - target_coords[1]) if screen_coords and target_coords else ''

    # 使用注视方向检测器
    gaze_angles, gaze_vector = gaze_detector.detect_gaze_direction(gaze_data)
    print(f"瞳孔角度: HR={gaze_data['hr'] if gaze_data['hr'] is not None else 'N/A'}, "
          f"VR={gaze_data['vr'] if gaze_data['vr'] is not None else 'N/A'}")
    print(f"合成向量: X={gaze_vector[0]:.2f}, Y={gaze_vector[1]:.2f}, Z={gaze_vector[2]:.2f}")

    # 获取滤波后的实时注视点坐标
    filtered_coords = gaze_data.get("filtered_screen_x"), gaze_data.get("filtered_screen_y")

    # 构建日志数据
    # 修改 example.py 中构建日志数据的部分，使用安全访问方式
    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "hr": gaze_data["hr"],
        "vr": gaze_data["vr"],
        "screen_x": screen_coords[0] if screen_coords else '',
        "screen_y": screen_coords[1] if screen_coords else '',
        "target_x": target_coords[0] if target_coords else '',
        "target_y": target_coords[1] if target_coords else '',
        "error_x": error_x,
        "error_y": error_y,
        "head_pitch": gaze_data["head_pitch"],
        "head_yaw": gaze_data["head_yaw"],
        "head_roll": gaze_data["head_roll"],
        "gaze_vx": gaze_data["gaze_vx"],
        "gaze_vy": gaze_data["gaze_vy"],
        "gaze_vz": gaze_data["gaze_vz"],
        "left_pupil_x": gaze_data["left_pupil_x"],
        "left_pupil_y": gaze_data["left_pupil_y"],
        "right_pupil_x": gaze_data["right_pupil_x"],
        "right_pupil_y": gaze_data["right_pupil_y"],
        "left_eye_inner_x": gaze_data.get("left_eye_inner_x", ''),
        "left_eye_inner_y": gaze_data.get("left_eye_inner_y", ''),
        "left_eye_outer_x": gaze_data.get("left_eye_outer_x", ''),
        "left_eye_outer_y": gaze_data.get("left_eye_outer_y", ''),
        "right_eye_inner_x": gaze_data.get("right_eye_inner_x", ''),
        "right_eye_inner_y": gaze_data.get("right_eye_inner_y", ''),
        "right_eye_outer_x": gaze_data.get("right_eye_outer_x", ''),
        "right_eye_outer_y": gaze_data.get("right_eye_outer_y", ''),
        "left_eye_upper_lid_x": gaze_data.get("left_eye_upper_lid_x", ''),
        "left_eye_upper_lid_y": gaze_data.get("left_eye_upper_lid_y", ''),
        "left_eye_lower_lid_x": gaze_data.get("left_eye_lower_lid_x", ''),
        "left_eye_lower_lid_y": gaze_data.get("left_eye_lower_lid_y", ''),
        "right_eye_upper_lid_x": gaze_data.get("right_eye_upper_lid_x", ''),
        "right_eye_upper_lid_y": gaze_data.get("right_eye_upper_lid_y", ''),
        "right_eye_lower_lid_x": gaze_data.get("right_eye_lower_lid_x", ''),
        "right_eye_lower_lid_y": gaze_data.get("right_eye_lower_lid_y", ''),
        "is_blinking": gaze_data["is_blinking"],
        "direction": gaze_data["direction"],
        "filtered_screen_x": filtered_coords[0] if filtered_coords[0] is not None else '',
        "filtered_screen_y": filtered_coords[1] if filtered_coords[1] is not None else '',
        "filtered_head_pitch": gaze_data["filtered_head_pose"][0] if "filtered_head_pose" in gaze_data and gaze_data[
            "filtered_head_pose"] else '',
        "filtered_head_yaw": gaze_data["filtered_head_pose"][1] if "filtered_head_pose" in gaze_data and gaze_data[
            "filtered_head_pose"] else '',
        "filtered_head_roll": gaze_data["filtered_head_pose"][2] if "filtered_head_pose" in gaze_data and gaze_data[
            "filtered_head_pose"] else ''
    }

    # 记录日志
    logger.log(log_data)
    logger.auto_flush()

    # 在追踪循环中获取 landmarks
    landmarks = None
    if hasattr(gaze, 'landmarks') and gaze.landmarks is not None:
        landmarks = gaze.landmarks.parts()

    # 使用 visualize_all 生成 annotated_frame
    annotated_frame = visualize_all(
        raw_frame,
        gaze_data,
        mode="tracking",
        landmarks=landmarks
    )

    # 使用自适应缩放函数
    resized_frame = resize_to_window(annotated_frame, 'Gaze Tracking', (screen_w, screen_h))

    # 写入视频帧
    logger.write_video_frame(screen_frame=resized_frame, user_frame=raw_frame)

    cv2.imshow('Gaze Tracking', resized_frame)

    # 添加帧率控制
    frame_start_time = control_frame_rate(frame_start_time)

    if cv2.waitKey(1) == 27:
        break

# --------------------------收尾阶段--------------------------
# 程序退出前刷新一次日志
logger.flush()

# 添加关键日志提取
print("[INFO] 正在提取关键日志信息...")
key_log_path = logger.extract_key_logs_on_exit()

# 释放视频写入器
logger.release_video_writers()
webcam.release()
cv2.destroyAllWindows()

# ====== 添加热力图生成代码 ======
if __name__ == "__main__":
    # 收集所有有效屏幕坐标
    screen_coords = [log[3:5] for log in logger.buffer if log[3] and log[4]]

    # 确保有数据
    if screen_coords:
        # 生成热力图
        heatmap = generate_heatmap(
            screen_coords,
            screen_size=get_screen_resolution(),
            output_path=os.path.join(logger.log_dir, "gaze_heatmap.jpg")
        )

        # 显示热力图（可选）
        cv2.imshow("Gaze Heatmap", heatmap)
        cv2.waitKey(3000)  # 显示3秒
        cv2.destroyAllWindows()
    else:
        print("No valid screen coordinates for heatmap.")