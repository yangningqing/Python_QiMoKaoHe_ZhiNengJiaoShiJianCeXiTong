
import os
from collections import deque
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from erweima import decode_qr_from_camera
from huanjingjiance import SensorSimulator
from kongzhiluoji import DEFAULT_ROOM, evaluate_controls, get_room_profile
from renlian_shibie import recognize_from_camera
from shujucunchu import (
    append_environment_record,
    append_sign_record,
    clear_sign_records,
    load_sign_names,
)

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


class SmartClassroomApp:

    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("智能教室环境管理系统")
        self.master.geometry("1100x820")
        self.simulator = SensorSimulator()
        self.current_room = DEFAULT_ROOM
        self.current_profile = get_room_profile(self.current_room)
        self.is_monitoring = False
        self.monitor_job = None
        self.history = deque(maxlen=50)
        base_dir = os.path.dirname(__file__)
        self.csv_path = os.path.join(base_dir, "classroom_data.csv")
        self.sign_csv_path = os.path.join(base_dir, "sign_records.csv")
        self.known_people = []
        self.camera_monitoring = False
        self.camera_job = None
        self.people_count = 0
        # 签到历史（格式：'时间  姓名'）
        self.sign_history: list[str] = load_sign_names(self.sign_csv_path)
        self.sign_dialog: tk.Toplevel | None = None
        self.sign_listbox: tk.Listbox | None = None

        self._build_ui()
        self._init_chart()

    def _build_ui(self):
        monitor_frame = ttk.LabelFrame(self.master, text="环境监测控制", padding=10)
        monitor_frame.pack(fill="x", padx=10, pady=(5, 0))

        ttk.Label(monitor_frame, text="当前教室：", font=("SimHei", 12)).pack(side="left")
        self.room_var = tk.StringVar(value=self.current_room)
        ttk.Label(monitor_frame, textvariable=self.room_var, font=("SimHei", 12, "bold")).pack(side="left", padx=(0, 20))

        ttk.Button(monitor_frame, text="开始监测", command=self.start_monitoring).pack(side="left", padx=5)
        ttk.Button(monitor_frame, text="停止监测", command=self.stop_monitoring).pack(side="left", padx=5)

        camera_frame = ttk.LabelFrame(self.master, text="人员检测（摄像头）", padding=10)
        camera_frame.pack(fill="x", padx=10, pady=(5, 0))
        ttk.Button(camera_frame, text="摄像头识别", command=self.recognize_camera).pack(side="left", padx=5)
        ttk.Button(camera_frame, text="开启人员检测", command=self.start_camera_monitor).pack(side="left", padx=5)
        ttk.Button(camera_frame, text="停止人员检测", command=self.stop_camera_monitor).pack(side="left", padx=5)
        ttk.Button(camera_frame, text="二维码签到", command=self.open_sign_dialog).pack(side="left", padx=5)

        status_frame = ttk.LabelFrame(self.master, text="实时环境数据", padding=10)
        status_frame.pack(fill="x", padx=10, pady=10)
        self.labels = {}
        line_items = [
            ("temperature", "温度(℃)"),
            ("light", "光照(lux)"),
            ("people", "人员数"),
        ]
        for col, (field, label) in enumerate(line_items):
            frame = ttk.Frame(status_frame, padding=5)
            frame.grid(row=0, column=col, sticky="w")
            ttk.Label(frame, text=f"{label}：", font=("SimHei", 11)).pack(side="left")
            var = tk.StringVar(value="--")
            ttk.Label(frame, textvariable=var, font=("Consolas", 12, "bold")).pack(side="left")
            self.labels[field] = var

        control_frame = ttk.LabelFrame(self.master, text="自动控制状态", padding=10)
        control_frame.pack(fill="x", padx=10)
        self.control_vars = {name: tk.StringVar(value="待机") for name in ["空调", "照明"]}
        for name, var in self.control_vars.items():
            ttk.Label(control_frame, text=f"{name}：", font=("SimHei", 11)).pack(side="left", padx=(10, 2))
            ttk.Label(control_frame, textvariable=var, font=("SimHei", 11, "bold")).pack(side="left", padx=5)

        chart_frame = ttk.LabelFrame(self.master, text="数据可视化（Matplotlib）", padding=10)
        chart_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.chart_frame = chart_frame

        people_frame = ttk.LabelFrame(self.master, text="教室内识别到的人员", padding=10)
        people_frame.pack(fill="x", padx=10, pady=5)
        self.people_list_var = tk.StringVar(value="暂无人员检测数据")
        ttk.Label(people_frame, textvariable=self.people_list_var, font=("SimHei", 11)).pack(side="left")
        self.people_count = 0
        self.labels["people"].set("0")

    def _init_chart(self):
        self.fig = Figure(figsize=(6, 3))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("温度/光照趋势")
        self.ax.set_xlabel("时间")
        self.ax.set_ylabel("数值")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def start_monitoring(self):
        if self.is_monitoring:
            return
        self.is_monitoring = True
        self._log("开始环境监测...")
        self._schedule_next()

    def stop_monitoring(self):
        self.is_monitoring = False
        if self.monitor_job is not None:
            self.master.after_cancel(self.monitor_job)
            self.monitor_job = None
        self._log("监测已暂停。")

    def _schedule_next(self):
        data = self.simulator.generate()
        data["people"] = self.people_count
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.history.append({"time": timestamp, **data})
        for key in ("temperature", "light"):
            self.labels[key].set(f"{data[key]}")

        controls = evaluate_controls(data, self.current_profile)
        for name, state in controls.items():
            self.control_vars[name].set(state)

        record = dict(data)
        record["people"] = self.people_count
        append_environment_record(self.csv_path, self.current_room, timestamp, record, controls)
        self._log(f"[{timestamp}] 数据：{data} 控制：{controls}")
        self.update_chart()

        if self.is_monitoring:
            self.monitor_job = self.master.after(2000, self._schedule_next)

    def update_chart(self):
        self.ax.clear()
        if not self.history:
            self.ax.set_title("温度/光照趋势")
            self.canvas.draw()
            return
        times = [item["time"] for item in self.history]
        temps = [item["temperature"] for item in self.history]
        self.ax.plot(times, temps, label="温度(℃)", color="tomato")
        lights = [item["light"] for item in self.history]
        self.ax.plot(times, lights, label="光照(lux)", color="goldenrod")
        self.ax.set_title("温度/光照趋势")
        self.ax.set_ylabel("数值")
        self.ax.set_xlabel("时间")
        self.ax.legend()
        self.ax.grid(alpha=0.2)
        self.fig.autofmt_xdate(rotation=45)
        self.canvas.draw()

    def _log(self, message: str):
        print(f"{datetime.now().strftime('%H:%M:%S')} {message}")

    def recognize_camera(self):
        self._log("启动摄像头人脸识别（按 q 关闭）...")
        try:
            identity, people_set = recognize_from_camera(duration_seconds=None, silent=False)
        except FileNotFoundError as err:
            messagebox.showerror("资源缺失", str(err))
            return
        except RuntimeError as err:
            messagebox.showerror("摄像头错误", str(err))
            return

        self._process_camera_result(identity, people_set, source="手动识别", notify=False)

    def start_camera_monitor(self):
        if self.camera_monitoring:
            return
        self.camera_monitoring = True
        self._log("人员检测已开启（每2秒静默识别）")
        self._camera_monitor_tick()

    def stop_camera_monitor(self):
        if not self.camera_monitoring:
            return
        self.camera_monitoring = False
        if self.camera_job is not None:
            self.master.after_cancel(self.camera_job)
            self.camera_job = None
        self._log("人员检测已停止")

    def _camera_monitor_tick(self):
        try:
            identity, people_set = recognize_from_camera(duration_seconds=1, silent=True)
        except Exception as err:  # pylint: disable=broad-exception-caught
            self._log(f"人员检测异常：{err}")
            self.stop_camera_monitor()
            return

        self._process_camera_result(identity, people_set, source="人员检测", notify=False)

        if self.camera_monitoring:
            self.camera_job = self.master.after(2000, self._camera_monitor_tick)

    def _process_camera_result(self, identity, people_set, source: str, notify: bool = False):
        if people_set:
            self.known_people = sorted(list(people_set))
            self.people_count = len(people_set)
            self.labels["people"].set(str(self.people_count))
            summary = f"{source}：识别到 {self.people_count} 人（{', '.join(self.known_people)}）"
            self.people_list_var.set(summary)
            self._log(summary)
            if notify:
                messagebox.showinfo("识别结果", summary)
            return

        self.known_people = []
        self.people_count = 0
        self.labels["people"].set("0")
        self.people_list_var.set(f"{source}：未检测到人员")
        if identity is None:
            self._log(f"{source}：未检测到人脸")
            if notify:
                messagebox.showinfo("识别结果", "未检测到人脸，请重试。")
            return

        self.people_count = 1
        self.labels["people"].set("1")
        if identity == "unknown":
            self._log(f"{source}：检测到未知人员")
            if notify:
                messagebox.showinfo("识别结果", "检测到未知人员（身份未登记）。")
        else:
            self._log(f"{source}：识别到 {identity}")
            if notify:
                messagebox.showinfo("识别结果", f"成功识别到 {identity}。")

    def open_sign_dialog(self):
        if self.sign_dialog and self.sign_dialog.winfo_exists():
            self.sign_dialog.lift()
            return

        dialog = tk.Toplevel(self.master)
        dialog.title("二维码签到")
        dialog.geometry("360x320")
        ttk.Label(dialog, text="请使用摄像头对准签到二维码，按 q 结束", font=("SimHei", 11)).pack(pady=5)
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="启动摄像头扫描", command=self.scan_qr_sign).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="清空签到记录", command=self.clear_sign_gui).pack(side="left", padx=5)

        self.sign_listbox = tk.Listbox(dialog, height=10)
        self.sign_listbox.pack(fill="both", expand=True, padx=10, pady=10)
        for item in self.sign_history:
            self.sign_listbox.insert("end", item)

        dialog.protocol("WM_DELETE_WINDOW", lambda: self._close_sign_dialog(dialog))
        self.sign_dialog = dialog

    def _close_sign_dialog(self, dialog: tk.Toplevel):
        if dialog and dialog.winfo_exists():
            dialog.destroy()
        self.sign_dialog = None
        self.sign_listbox = None

    def clear_sign_gui(self):
        if messagebox.askyesno("确认清空", "确定要清空所有签到记录吗？此操作不可恢复。"):
            clear_sign_records(self.sign_csv_path)
            self.sign_history.clear()
            if self.sign_listbox and self.sign_listbox.winfo_exists():
                self.sign_listbox.delete(0, "end")

    def scan_qr_sign(self):
        try:
            content = decode_qr_from_camera()
        except Exception as err:  # pylint: disable=broad-exception-caught
            messagebox.showerror("识别异常", f"二维码识别失败：{err}")
            return

        if not content:
            messagebox.showwarning("识别失败", "未检测到有效二维码内容。")
            return

        student_name = content.strip()
        append_sign_record(self.sign_csv_path, student_name)
        # 使用当前时间与姓名组成一条可读记录
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record_line = f"{time_str}  {student_name}"
        self.sign_history.append(record_line)
        if self.sign_listbox and self.sign_listbox.winfo_exists():
            self.sign_listbox.insert("end", record_line)
            self.sign_listbox.see("end")
        self._log(f"签到成功：{student_name}")
        messagebox.showinfo("签到成功", f"{student_name} 签到成功！")

    def on_close(self):
        """窗口关闭时的统一处理：停止监测与人员检测并销毁主窗口。"""
        try:
            self.stop_camera_monitor()
        except Exception:
            pass
        try:
            self.stop_monitoring()
        except Exception:
            pass
        if self.sign_dialog and self.sign_dialog.winfo_exists():
            self.sign_dialog.destroy()
        self.master.destroy()


def main():
    root = tk.Tk()
    app = SmartClassroomApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()

