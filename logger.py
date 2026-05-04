# logger.py
import csv
import os
import time
from datetime import datetime
from collections import deque

import cv2


class GazeLogger:
    def __init__(self, output_dir="output", user_info=None):
        # 如果提供了用户信息，则按用户ID和姓名创建目录结构
        if user_info:
            user_id = user_info.get('id', 'unknown')
            user_name = user_info.get('original_name', 'unknown')
            date_str = datetime.now().strftime("%Y-%m-%d")
            subject_dir = f"{user_id}_{user_name}_{date_str}"
            self.output_dir = os.path.join(os.path.abspath(output_dir), subject_dir)
        else:
            self.output_dir = os.path.abspath(output_dir)

        # 创建主目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 创建子目录
        self.videos_dir = os.path.join(self.output_dir, "videos")
        self.logs_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # 直接在logs目录下创建带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.logs_dir, f"gaze_log_{timestamp}.csv")
        self.buffer = deque()
        self.last_flush_time = time.time()
        self.LOG_FLUSH_INTERVAL = 5

        # 保存用户信息
        self.user_info = user_info

        # 如果有用户信息，保存用户信息到subject_info.csv
        if self.user_info:
            self._save_subject_info()
            self._update_subject_total_info()

        # 视频写入器相关属性
        self.video_writers = {}  # 用于存储不同阶段的视频写入器
        self.current_stage = None
        self.video_timestamp = timestamp  # 固定时间戳用于所有视频文件

        # 视频参数
        self.fps = 30  # 固定帧率
        self.frame_count = {}  # 记录每个视频的帧数

        # 写入新表头
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "horizontal_ratio", "vertical_ratio",
                "screen_x", "screen_y",  # 用户实际注视点
                "target_x", "target_y",  # 参考点坐标
                "error_x", "error_y",  # 两个坐标的差值
                "head_pitch", "head_yaw", "head_roll",
                "gaze_vx", "gaze_vy", "gaze_vz",
                "is_blinking",
                "direction",
                "left_pupil_x", "left_pupil_y",
                "right_pupil_x", "right_pupil_y",
                "left_eye_inner_x", "left_eye_inner_y",
                "left_eye_outer_x", "left_eye_outer_y",
                "right_eye_inner_x", "right_eye_inner_y",
                "right_eye_outer_x", "right_eye_outer_y",
                "left_eye_upper_lid_x", "left_eye_upper_lid_y",
                "left_eye_lower_lid_x", "left_eye_lower_lid_y",
                "right_eye_upper_lid_x", "right_eye_upper_lid_y",
                "right_eye_lower_lid_x", "right_eye_lower_lid_y",
                "filtered_screen_x", "filtered_screen_y",
                "filtered_head_pitch", "filtered_head_yaw", "filtered_head_roll"
            ])

    def _save_subject_info(self):
        """保存参与者信息到subject_info.csv"""
        subject_info_file = os.path.join(self.output_dir, "subject_info.csv")
        with open(subject_info_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "original_name", "name", "gender", "age",
                "vision_correction", "dominant_eye", "screen_time_hrs",
                "sleep_hrs", "medication", "notes"
            ])
            writer.writerow([
                self.user_info.get('id', ''),
                self.user_info.get('original_name', ''),
                self.user_info.get('name', ''),
                self.user_info.get('gender', ''),
                self.user_info.get('age', ''),
                self.user_info.get('vision_correction', ''),
                self.user_info.get('dominant_eye', ''),
                self.user_info.get('screen_time_hrs', ''),
                self.user_info.get('sleep_hrs', ''),
                self.user_info.get('medication', ''),
                self.user_info.get('notes', '')
            ])

    def _update_subject_total_info(self):
        """更新总参与者信息文件"""
        total_info_file = os.path.join(os.path.dirname(self.output_dir), "subject_total_info.csv")
        file_exists = os.path.exists(total_info_file)

        with open(total_info_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "id", "original_name", "name", "gender", "age",
                    "vision_correction", "dominant_eye", "screen_time_hrs",
                    "sleep_hrs", "medication", "notes", "date"
                ])
            writer.writerow([
                self.user_info.get('id', ''),
                self.user_info.get('original_name', ''),
                self.user_info.get('name', ''),
                self.user_info.get('gender', ''),
                self.user_info.get('age', ''),
                self.user_info.get('vision_correction', ''),
                self.user_info.get('dominant_eye', ''),
                self.user_info.get('screen_time_hrs', ''),
                self.user_info.get('sleep_hrs', ''),
                self.user_info.get('medication', ''),
                self.user_info.get('notes', ''),
                datetime.now().strftime("%Y-%m-%d")
            ])

    def init_video_writers(self, screen_frame_shape, user_frame_shape, fps=30):
        """
        初始化视频写入器
        :param screen_frame_shape: 屏幕帧的形状 (height, width, channels)
        :param user_frame_shape: 用户帧的形状 (height, width, channels)
        :param fps: 视频帧率
        """
        # 确保帧形状有效
        if screen_frame_shape is None or user_frame_shape is None:
            print("[WARN] 无法初始化视频写入器：帧形状无效")
            return

        screen_height, screen_width = screen_frame_shape[:2]
        user_height, user_width = user_frame_shape[:2]

        # 验证尺寸有效性
        if screen_width <= 0 or screen_height <= 0 or user_width <= 0 or user_height <= 0:
            print("[WARN] 无法初始化视频写入器：帧尺寸无效")
            return

        # 使用实际摄像头的帧率而不是固定30FPS
        self.fps = fps
        self.frame_count = {}  # 重置帧计数

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 为不同阶段创建视频文件
        stages = ['calibration', 'validation', 'tracking']
        for stage in stages:
            # 屏幕视频
            screen_video_path = os.path.join(self.videos_dir, f"{stage}_screen_{timestamp}.mp4")
            self.video_writers[f"{stage}_screen"] = cv2.VideoWriter(
                screen_video_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,  # 使用传入的fps参数
                (screen_width, screen_height)
            )

            # 初始化帧计数
            self.frame_count[f"{stage}_screen"] = 0

            # 用户视频
            user_video_path = os.path.join(self.videos_dir, f"{stage}_user_{timestamp}.mp4")
            self.video_writers[f"{stage}_user"] = cv2.VideoWriter(
                user_video_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,  # 使用传入的fps参数
                (user_width, user_height)
            )

            # 初始化帧计数
            self.frame_count[f"{stage}_user"] = 0

            print(f"[INFO] 初始化视频写入器: {stage} 阶段，帧率: {fps} FPS")

    def set_video_fps(self, fps):
        """
        设置视频帧率
        :param fps: 帧率
        """
        self.fps = fps
        print(f"[INFO] 视频帧率设置为: {fps} FPS")

    def set_current_stage(self, stage):
        """
        设置当前阶段
        :param stage: 阶段名称 ('calibration', 'validation', 'tracking')
        """
        if stage in ['calibration', 'validation', 'tracking']:
            self.current_stage = stage
            print(f"[INFO] 设置当前阶段为: {stage}")
        else:
            print(f"[WARN] 无效的阶段名称: {stage}")

    def write_video_frame(self, screen_frame=None, user_frame=None):
        """
        写入视频帧到当前阶段的视频文件
        :param screen_frame: 屏幕帧
        :param user_frame: 用户帧
        """
        if self.current_stage is None:
            return

        try:
            if screen_frame is not None and f"{self.current_stage}_screen" in self.video_writers:
                writer = self.video_writers[f"{self.current_stage}_screen"]
                if writer and writer.isOpened():
                    # 确保帧尺寸正确
                    writer.write(screen_frame)
                    self.frame_count[f"{self.current_stage}_screen"] += 1

            if user_frame is not None and f"{self.current_stage}_user" in self.video_writers:
                writer = self.video_writers[f"{self.current_stage}_user"]
                if writer and writer.isOpened():
                    # 确保帧尺寸正确
                    writer.write(user_frame)
                    self.frame_count[f"{self.current_stage}_user"] += 1

        except Exception as e:
            print(f"[ERROR] 写入视频帧时出错: {e}")

    def release_video_writers(self):
        """释放所有视频写入器"""
        print("[INFO] 正在释放视频写入器...")
        for key, writer in self.video_writers.items():
            if writer is not None:
                try:
                    frame_count = self.frame_count.get(key, 0)
                    print(f"[INFO] 视频 {key} 写入了 {frame_count} 帧")
                    writer.release()
                except Exception as e:
                    print(f"[ERROR] 释放视频写入器 {key} 时出错: {e}")

        self.video_writers.clear()
        self.current_stage = None
        print("[INFO] 视频写入器已释放")

    def log(self, data):
        if isinstance(data, dict):
            self._log_dict(data)
        elif isinstance(data, (list, tuple)):
            self.buffer.append(data)
        else:
            raise ValueError("Unsupported log data type.")

    def _log_dict(self, data):
        log_entry = [
            data.get('timestamp', ''),
            data.get('hr', ''),  # 水平注释比例
            data.get('vr', ''),
            data.get('screen_x', ''),  # 修改字段名以匹配 gaze_data
            data.get('screen_y', ''),
            data.get('target_x', ''),
            data.get('target_y', ''),
            data.get('error_x', ''),
            data.get('error_y', ''),
            data.get('head_pitch', ''),  # 头部姿态角度
            data.get('head_yaw', ''),
            data.get('head_roll', ''),
            data.get('gaze_vx', ''),  # 3D注视向量
            data.get('gaze_vy', ''),
            data.get('gaze_vz', ''),
            data.get('is_blinking', ''),
            data.get('direction', ''),
            data.get('left_pupil_x', ''),
            data.get('left_pupil_y', ''),
            data.get('right_pupil_x', ''),
            data.get('right_pupil_y', ''),
            data.get('left_eye_inner_x', ''),
            data.get('left_eye_inner_y', ''),
            data.get('left_eye_outer_x', ''),
            data.get('left_eye_outer_y', ''),
            data.get('right_eye_inner_x', ''),
            data.get('right_eye_inner_y', ''),
            data.get('right_eye_outer_x', ''),
            data.get('right_eye_outer_y', ''),
            data.get('left_eye_upper_lid_x', ''),
            data.get('left_eye_upper_lid_y', ''),
            data.get('left_eye_lower_lid_x', ''),
            data.get('left_eye_lower_lid_y', ''),
            data.get('right_eye_upper_lid_x', ''),
            data.get('right_eye_upper_lid_y', ''),
            data.get('right_eye_lower_lid_x', ''),
            data.get('right_eye_lower_lid_y', ''),
            data.get('filtered_screen_x', ''),
            data.get('filtered_screen_y', ''),
            data.get('filtered_head_pitch', ''),
            data.get('filtered_head_yaw', ''),
            data.get('filtered_head_roll', '')
        ]
        print("Log entry to be written:", log_entry)  # 调试输出
        self.buffer.append(log_entry)

    def flush(self):
        if len(self.buffer) > 0:
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(self.buffer)
            self.buffer.clear()
        self.last_flush_time = time.time()

    def auto_flush(self):
        if time.time() - self.last_flush_time >= self.LOG_FLUSH_INTERVAL:
            self.flush()

    @property
    def log_path(self):
        return self.log_file

    @property
    def log_dir(self):
        return self.logs_dir

    def log_error(self, message):
        # 错误日志也放在logs目录下
        error_log_path = os.path.join(self.logs_dir, "error_log.txt")
        with open(error_log_path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
        print(f"[ERROR] {message}")  # 同时输出到控制台

    # 在 logger.py 中添加以下方法到 GazeLogger 类

    def extract_key_logs_on_exit(self):
        """
        在程序退出时自动提取关键日志
        """
        import re

        try:
            # 生成关键日志文件路径
            base_name = os.path.splitext(self.log_file)[0]
            key_log_path = f"{base_name}_key_log.txt"

            # 要提取的关键阶段
            key_phrases = ["calibration_point", "validation_point"]

            with open(self.log_file, "r", encoding="utf-8") as f:
                full_log = f.readlines()

            key_lines = []

            # 读取表头以确定字段位置
            if full_log:
                headers = full_log[0].strip().split(',')

                # 关键字段索引
                key_indices = {}
                key_fields = ["timestamp", "target_x", "target_y", "screen_x", "screen_y",
                              "error_x", "error_y", "head_pitch", "head_yaw", "horizontal_ratio", "vertical_ratio",
                              "direction"]

                for field in key_fields:
                    try:
                        key_indices[field] = headers.index(field)
                    except ValueError:
                        # 如果找不到字段，尝试匹配相似名称
                        for i, header in enumerate(headers):
                            if field in header or header in field:
                                key_indices[field] = i
                                break

                # 处理数据行
                for line in full_log[1:]:  # 跳过表头
                    items = line.strip().split(',')
                    if len(items) < len(headers):
                        continue

                    # 检查是否包含关键阶段
                    direction = items[key_indices.get("direction", -1)] if "direction" in key_indices else ""
                    if not any(phrase in direction for phrase in key_phrases):
                        continue

                    # 提取关键字段
                    log_data = {}
                    for field, index in key_indices.items():
                        if index < len(items):
                            log_data[field] = items[index]

                    if log_data:
                        stage = '校准' if 'calibration' in log_data.get('direction', '') else '验证'
                        key_lines.append(f"阶段: {stage}, " +
                                         ", ".join([f"{k}:{v}" for k, v in log_data.items()]) + "\n")

            # 保存提取的关键日志
            with open(key_log_path, "w", encoding="utf-8") as f:
                f.writelines(key_lines)
            print(f"[INFO] 关键日志已保存到 {key_log_path}，共 {len(key_lines)} 行")
            return key_log_path
        except Exception as e:
            print(f"[ERROR] 提取关键日志时出错: {e}")
            return None

