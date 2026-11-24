"""CLI 入口。当前为骨架,交互设计见 REQ.md 待定项。"""
import sys


def main() -> None:
    # Windows 控制台默认 GBK,强制 UTF-8 输出,避免中文乱码
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    print("码猫 CodeCat v0.1.0 —— 开发中,先跑 scripts/demo.py 验证环境")


if __name__ == "__main__":
    main()
