
from typing import Dict, Tuple

ROOM_PROFILES: Dict[str, Dict[str, Tuple[int, int]]] = {
    "A-101": {"temp_range": (22, 26), "light_range": (350, 500)},

}

DEFAULT_ROOM = "A-101"


def get_room_profile(room_id: str) -> Dict[str, Tuple[int, int]]:
    return ROOM_PROFILES.get(room_id, ROOM_PROFILES[DEFAULT_ROOM])


def evaluate_controls(data: Dict[str, float], profile: Dict[str, Tuple[int, int]]) -> Dict[str, str]:
    temp_min, temp_max = profile["temp_range"]
    light_min, light_max = profile["light_range"]

    controls = {}
    if data["temperature"] > temp_max:
        controls["空调"] = "制冷中"
    elif data["temperature"] < temp_min:
        controls["空调"] = "制热中"
    else:
        controls["空调"] = "待机"

    if data["light"] < light_min and data["people"] > 0:
        controls["照明"] = "开灯"
    elif data["light"] > light_max:
        controls["照明"] = "调暗"
    else:
        controls["照明"] = "维持"

    return controls


if __name__ == "__main__":
    sample = {"temperature": 28, "light": 200, "people": 1}
    print(evaluate_controls(sample, get_room_profile("A-101")))

