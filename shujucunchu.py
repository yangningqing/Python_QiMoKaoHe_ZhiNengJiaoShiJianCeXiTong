
import csv
import os
from datetime import datetime

FIELDNAMES = ["时间", "教室", "温度", "光照", "人员数", "空调", "照明"]
SIGN_FIELDNAMES = ["时间", "姓名", "来源"]


def append_environment_record(csv_path, room, timestamp, data, controls):
    """将监测数据追加写入 CSV 文件。"""
    file_exists = os.path.exists(csv_path)
    row = [
        timestamp,
        room,
        data["temperature"],
        data["light"],
        data["people"],
        controls["空调"],
        controls["照明"],
    ]
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(FIELDNAMES)
        writer.writerow(row)


def append_sign_record(csv_path, student_name, source="二维码"):
    """记录学生签到信息。"""
    file_exists = os.path.exists(csv_path)
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        student_name,
        source,
    ]
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(SIGN_FIELDNAMES)
        writer.writerow(row)

def load_sign_names(csv_path):
    """
    从签到 CSV 中加载历史记录，格式为 '时间 姓名' 的字符串列表。
    """
    if not os.path.exists(csv_path):
        return []
    records = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            next(reader)  # 跳过表头
        except StopIteration:
            return []
        for row in reader:
            if len(row) >= 2:
                time_str, name = row[0], row[1]
                records.append(f"{time_str}  {name}")
    return records
def clear_sign_records(csv_path):
    if os.path.exists(csv_path):
        os.remove(csv_path)


if __name__ == "__main__":
    demo_data = {"temperature": 25, "light": 350, "people": 1}
    demo_control = {"空调": "制冷中", "照明": "开灯"}
    append_environment_record("demo.csv", "A-101", "12:00:00", demo_data, demo_control)
    append_sign_record("sign_demo.csv", "张三")
    print("已生成 demo.csv 与 sign_demo.csv")

