"""Operator task-text prompt shared by the live entries."""

from ..aubo_joint_contract import JOINT_TASK


def read_task():
    print(f'请输入本轮任务指令（输入 q 退出）：\n{JOINT_TASK}', flush=True)
    while True:
        task = input('任务指令：').strip()
        if task.lower() == 'q':
            return None
        if task == JOINT_TASK:
            return task
        print('当前模型仅训练了上面这句任务，请完整输入或粘贴；输入 q 退出。', flush=True)
