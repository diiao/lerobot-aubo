#!/usr/bin/env python
"""AUBO 数字输出只读检查脚本。

用法：
    python examples/phone_to_auboi10/test_io.py

本脚本不会写入任何输出。禁止通过依次拉高全部输出的方式查找夹爪端口，
因为其他端口可能连接未识别的执行器。夹爪已知端口应通过受控适配器
单独验证。
"""

import pyaubo_sdk

ROBOT_IP = "192.168.31.200"
ROBOT_PORT = 30004


def main():
    client = pyaubo_sdk.RpcClient()
    client.setRequestTimeout(1000)
    client.connect(ROBOT_IP, ROBOT_PORT)
    if not client.hasConnected():
        print("连接失败！请检查机械臂 IP 和端口。")
        return

    try:
        client.login("aubo", "123456")
        if not client.hasLogined():
            print("登录失败！")
            return

        robot_name = client.getRobotNames()[0]
        io = client.getRobotInterface(robot_name).getIoControl()
        std_out_num = io.getStandardDigitalOutputNum()
        tool_out_num = io.getToolDigitalOutputNum()
        print(f"标准数字输出数量: {std_out_num}")
        print(f"工具端数字输出数量: {tool_out_num}")

        print("\n当前标准数字输出状态:")
        for i in range(std_out_num):
            print(f"  标准输出[{i}] = {io.getStandardDigitalOutput(i)}")

        print("\n当前工具端数字输出状态:")
        for i in range(tool_out_num):
            print(f"  工具端输出[{i}] = {io.getToolDigitalOutput(i)}")

        print("\n只读检查完成；未修改任何数字输出。")
        print("已知夹爪命令映射：标准 DO2=闭合，标准 DO3=张开，两路必须互斥。")
    finally:
        if client.hasLogined():
            client.logout()
        if client.hasConnected():
            client.disconnect()


if __name__ == "__main__":
    main()
