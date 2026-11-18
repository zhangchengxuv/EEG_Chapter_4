"""一键执行独立实验：ADS1299 从头训练，然后进行单独保存的 g.Nautilus 外部复测。"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent


def run_with_log(command, log_path):
    """同时显示与保存完整日志；子步骤失败就停止，避免误报整轮完成。"""
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        status = process.wait()
    if status != 0:
        raise RuntimeError(f"子步骤退出码为 {status}，请检查日志：{log_path}")


def main():
    """每次创建新的实验名，不覆盖已训练好的模型或结果。"""
    parser = argparse.ArgumentParser(description="独立重跑选定 EEGNet 并进行跨设备外部复测")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    name = args.run_name or datetime.now(ZoneInfo("Asia/Shanghai")).strftime("run_%Y%m%d_%H%M%S")
    if Path(name).name != name:
        parser.error("run-name 只能是目录名称")
    if (ROOT / "results" / name).exists() or (ROOT / "models" / name).exists():
        raise FileExistsError(f"实验名称已存在，不覆盖：{name}")
    # 使用启动本脚本的同一解释器运行两个步骤，不依赖旧实验目录或其虚拟环境。
    for script, label in [("train.py", "ADS1299重新训练"), ("test_gnautilus.py", "gNautilus外部复测")]:
        print(f"\n========== {label} ==========", flush=True)
        command = [sys.executable, "-u", str(ROOT / "code" / script), "--run-name", name, "--device", args.device]
        run_with_log(command, ROOT / "logs" / f"{name}_{label}.log")
    (ROOT / "latest_run.txt").write_text(name, encoding="utf-8")
    print(f"\n本轮全部完成：{ROOT / 'results' / name}", flush=True)


if __name__ == "__main__":
    main()
