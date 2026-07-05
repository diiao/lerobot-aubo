#!/usr/bin/env python
"""
IO 扫描测试脚本
逐个打开每个数字输出口（标准 + 工具端），每次 1.5 秒，方便确认吸盘接的是哪个端口。

用法：
    python examples/phone_to_auboi10/test_io.py

按 Ctrl+C 随时中止。脚本结束后所有输出自动关闭。
"""

import time
import pyaubo_sdk

ROBOT_IP = "192.168.31.200"
ROBOT_PORT = 30004
ON_DURATION = 1.5   # 每个引脚保持打开的时间(秒)
OFF_DURATION = 0.5  # 关闭间隔(秒)


def main():
    client = pyaubo_sdk.RpcClient()
    client.setRequestTimeout(1000)
    client.connect(ROBOT_IP, ROBOT_PORT)
    if not client.hasConnected():
        print("连接失败！请检查机械臂 IP 和端口。")
        return

    client.login("aubo", "123456")
    if not client.hasLogined():
        print("登录失败！")
        return

    robot_name = client.getRobotNames()[0]
    io = client.getRobotInterface(robot_name).getIoControl()

    std_out_num = io.getStandardDigitalOutputNum()
    tool_out_num = io.getToolDigitalOutputNum()

    print(f"\n{'='*50}")
    print(f"标准数字输出数量: {std_out_num}")
    print(f"工具端数字输出数量: {tool_out_num}")

    # 读取当前输出状态
    print("\n当前标准数字输出状态:")
    for i in range(std_out_num):
        val = io.getStandardDigitalOutput(i)
        print(f"  标准输出[{i}] = {val}")

    print("\n当前工具端数字输出状态:")
    for i in range(tool_out_num):
        val = io.getToolDigitalOutput(i)
        print(f"  工具端输出[{i}] = {val}")

    print(f"\n{'='*50}")
    print("开始逐个测试每个输出口...")
    print("观察吸盘动作，记录哪个编号触发了吸盘。")
    print("按 Ctrl+C 中止。\n")

    def safe_off_all():
        for i in range(std_out_num):
            try:
                io.setStandardDigitalOutput(i, False)
            except Exception:
                pass
        for i in range(tool_out_num):
            try:
                io.setToolDigitalOutput(i, False)
            except Exception:
                pass

    try:
        # 测试标准数字输出
        for i in range(std_out_num):
            print(f">>> 标准数字输出 [{i}] ON  ({ON_DURATION}s)...")
            io.setStandardDigitalOutput(i, True)
            time.sleep(ON_DURATION)
            io.setStandardDigitalOutput(i, False)
            print(f"    标准数字输出 [{i}] OFF")
            time.sleep(OFF_DURATION)

        # 测试工具端数字输出
        for i in range(tool_out_num):
            print(f">>> 工具端数字输出 [{i}] ON  ({ON_DURATION}s)...")
            try:
                io.setToolDigitalOutput(i, True)
                time.sleep(ON_DURATION)
                io.setToolDigitalOutput(i, False)
                print(f"    工具端数字输出 [{i}] OFF")
            except Exception as e:
                print(f"    工具端输出 [{i}] 异常: {e}")
            time.sleep(OFF_DURATION)

        print("\n扫描完成！记录下触发吸盘的端口类型和编号。")

    except KeyboardInterrupt:
        print("\n中止，正在关闭所有输出...")
    finally:
        safe_off_all()
        print("所有输出已关闭。")
        client.logout()
        client.disconnect()


if __name__ == "__main__":
    main()
