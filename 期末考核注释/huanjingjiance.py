
import random
from typing import Dict

class SensorSimulator:
    """生成温度、光照等模拟数据，人员数由摄像头检测模块提供。"""

    def __init__(self):
        self.base_temp = 24
        self.base_light = 400

    def generate(self) -> Dict[str, float]:
        fluctuation = lambda base, delta: round(random.uniform(base - delta, base + delta), 1)
        return {
            "temperature": fluctuation(self.base_temp, 3),
            "light": round(random.uniform(self.base_light - 150, self.base_light + 150), 0),
        }


if __name__ == "__main__":
    sim = SensorSimulator()
    for _ in range(3):
        print(sim.generate())

