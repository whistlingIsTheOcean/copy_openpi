"""部署验证 · 第 3 步：不动机器人，验证「网络 + 序列化 + 模型推理」这条链路。

两种用法
--------
A. 纯离线（不需要机器人 / 相机，只需要 openpi_client + numpy）
     python examples/our_real/dry_run.py --host 172.16.1.3 --port 8001

B. 只读真实硬件（读真实 TCP 位姿 + 真实相机图，但**绝不发送任何动作**）
     python -m examples.our_real.dry_run --host 172.16.1.3 --port 8001 --use-robot

B 用于验证「真实观测 → 模型输入」这一段（相机图尺寸/dtype、tcp→rotation_6d、
夹爪归一化），全程不调用 apply_action，所以机器人不会动。

退出码 0 = 全部检查通过。
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import socket
import sys
import time

import numpy as np

from openpi_client import action_chunk_broker
from openpi_client import msgpack_numpy as _msgpack_numpy
from openpi_client import websocket_client_policy as _wcp

# ---- 以下常量必须与真实部署保持一致 ----
IMG_SHAPE = (720, 1280, 3)
PROMPT = "Pick up the colored cup and move it into the metal cup"
ACTION_DIM = 10
# ready_pose = [0.4, 0.0, 0.25] + ready_rot_6d = [-1, 0, 0, 0, 1, 0] + gripper
READY_STATE = np.array([0.4, 0.0, 0.25, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
SAFE_MIN = np.array([0.2, -0.4, 0.0])  # 与 utils/constants.py 的 SAFE_WORKSPACE_MIN 一致
SAFE_MAX = np.array([0.8, 0.4, 0.4])   # 与 utils/constants.py 的 SAFE_WORKSPACE_MAX 一致
GRIPPER_MAX = 0.095                    # 与 decode_gripper_width 的 /1000*0.095 一致


def preflight(host: str, port: int, timeout: float = 5.0) -> tuple[bool, str]:
    """快速探测 TCP 端口。

    WebsocketClientPolicy._wait_for_server() 在 ConnectionRefusedError 时每 5 秒重试
    且永不放弃，所以服务端没起时它会一直挂着。这里先自己探一下，好给出明确报错。
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, ""
    except OSError as e:
        return False, f"{type(e).__name__}: {e}"


def make_offline_obs(i: int) -> dict:
    """纯离线假观测：让 state 的 x 轻微漂移，便于观察模型是否真的在响应。"""
    state = READY_STATE.copy()
    state[0] += 0.002 * i
    return {
        "observation/state": state,
        "observation/image": np.zeros(IMG_SHAPE, dtype=np.uint8),
        "observation/wrist_image": np.zeros(IMG_SHAPE, dtype=np.uint8),
        "prompt": PROMPT,
    }


def check_obs(obs: dict) -> bool:
    ok = True
    for key in ("observation/state", "observation/image", "observation/wrist_image", "prompt"):
        if key not in obs:
            print(f"    [FAIL] 观测缺少 key '{key}'（当前 keys: {sorted(obs)}）")
            ok = False
    if not ok:
        return False

    state = np.asarray(obs["observation/state"])
    img = np.asarray(obs["observation/image"])
    wrist = np.asarray(obs["observation/wrist_image"])

    if state.shape != (ACTION_DIM,):
        print(f"    [FAIL] observation/state 形状 {state.shape}，期望 ({ACTION_DIM},)")
        ok = False
    if img.shape != IMG_SHAPE:
        print(f"    [FAIL] observation/image 形状 {img.shape}，期望 {IMG_SHAPE}")
        ok = False
    if wrist.shape != IMG_SHAPE:
        print(f"    [FAIL] observation/wrist_image 形状 {wrist.shape}，期望 {IMG_SHAPE}")
        ok = False
    if img.dtype != np.uint8:
        print(f"    [FAIL] observation/image dtype {img.dtype}，期望 uint8")
        ok = False
    if not np.isfinite(state).all():
        print("    [FAIL] observation/state 里有 NaN / Inf")
        ok = False

    if ok:
        print(
            f"    [ok] state{state.shape}  image{img.shape}/{img.dtype}  "
            f"wrist{wrist.shape}/{wrist.dtype}  prompt={obs['prompt']!r}"
        )
    return ok


def check_actions(action, label: str) -> bool:
    a = np.asarray(action)
    if a.ndim != 2 or a.shape[-1] != ACTION_DIM:
        print(f"    [FAIL] {label}: 动作形状 {a.shape}，期望 (H, {ACTION_DIM})")
        return False

    ok = True
    if not np.isfinite(a).all():
        print(f"    [FAIL] {label}: 动作里有 NaN / Inf")
        ok = False
    if not np.any(np.abs(a[:, :3]) > 1e-6):
        print(f"    [FAIL] {label}: xyz 全为 0，模型没有在响应")
        ok = False

    xyz = a[:, :3]
    if xyz.min() < SAFE_MIN.min() - 0.05 or xyz.max() > SAFE_MAX.max() + 0.05:
        print(
            f"    [warn] {label}: xyz 超出安全区 [{SAFE_MIN.min():.2f}, {SAFE_MAX.max():.2f}]，"
            f"实际 [{xyz.min():.3f}, {xyz.max():.3f}]"
        )

    # rot6d 是旋转矩阵的前两行 => 每行范数应为 1
    rows = a[:, 3:9].reshape(len(a), 2, 3)
    norms = np.linalg.norm(rows, axis=-1)
    if not np.allclose(norms, 1.0, atol=0.05):
        print(
            f"    [warn] {label}: rot6d 行范数偏离 1（min={norms.min():.3f} max={norms.max():.3f}），"
            "旋转表示可能有问题"
        )
        ok = False

    grip = a[:, 9]
    if grip.min() < -0.05 or grip.max() > GRIPPER_MAX + 0.02:
        print(
            f"    [warn] {label}: gripper 超出 [0, {GRIPPER_MAX}] m，"
            f"实际 [{grip.min():+.4f}, {grip.max():+.4f}]"
        )

    print(
        f"    [ok] {label}: shape={a.shape}  "
        f"xyz∈[{xyz.min():+.3f}, {xyz.max():+.3f}]  "
        f"rot6d行范数∈[{norms.min():.3f}, {norms.max():.3f}]  "
        f"gripper∈[{grip.min():+.4f}, {grip.max():+.4f}]"
    )
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", required=True, help="服务端 IP，填纯 IP（不要带 ws://，不要填 0.0.0.0）")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--action-horizon", type=int, default=25, help="ActionChunkBroker 每次缓存的 chunk 长度")
    parser.add_argument("--num-steps", type=int, default=30, help="broker 连跑多少步")
    parser.add_argument("--bench-runs", type=int, default=5, help="裸 infer 计时次数")
    parser.add_argument("--use-robot", action="store_true", help="读真实硬件观测（不会发送任何动作）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)

    print("=" * 70)
    print("部署验证 · 第 3 步：网络 + 序列化 + 模型推理 链路检查")
    print("=" * 70)

    # ---- [1/5] 连通性 ----
    print(f"\n[1/5] TCP 连通性 {args.host}:{args.port}")
    reachable, err = preflight(args.host, args.port)
    if not reachable:
        print(f"    [FAIL] {err}")
        print("    排查清单：")
        print("      1) 服务端起了吗？   在服务端看有没有 serve_policy 进程")
        print(f"      2) 端口对吗？       在服务端 curl http://127.0.0.1:{args.port}/healthz")
        print(f"      3) 本机网卡通吗？   在服务端 curl http://<本机IP>:{args.port}/healthz")
        print("      4) IP 选对了吗？    服务端跑 `ip route get <本机IP>` 看走哪个网卡")
        print("      5) 防火墙开了吗？   sudo ufw status")
        return 1
    print("    [ok] 端口可达")

    t0 = time.perf_counter()
    policy = _wcp.WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"    [ok] websocket 握手成功（{time.perf_counter() - t0:.2f}s）")
    print(f"    服务端 metadata: {policy.get_server_metadata()}")

    # ---- [2/5] 观测 ----
    robot_env = None
    if args.use_robot:
        print("\n[2/5] 读取真实硬件观测（只读，不会调用 apply_action）")
        root = pathlib.Path(__file__).resolve().parents[2]  # copy_openpi/
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from examples.our_real import env as _env  # 延迟导入：离线模式不需要机器人依赖

        robot_env = _env.FlexivRealEnvironment(discretize_rotation=False)
        print("    [ok] 机器人 + 相机已连接")
    else:
        print("\n[2/5] 构造离线假观测（未加 --use-robot）")

    obs = robot_env.get_observation() if robot_env is not None else make_offline_obs(0)
    if not check_obs(obs):
        return 1

    # ---- [3/5] 裸 infer 计时 ----
    print("\n[3/5] 裸 infer 计时（网络往返 + 服务端推理）")

    # 先量一下客户端打包开销和传输体积。图像是两张 720x1280x3，体积不小，
    # 如果局域网只有千兆，光传输就可能吃掉几十毫秒。
    payload = _msgpack_numpy.Packer().pack(obs)
    print(f"    obs 体积 {len(payload) / 1e6:.2f} MB（千兆网单程约需 {len(payload) * 8 / 1e6:.0f} ms）")

    try:
        t0 = time.perf_counter()
        policy.infer(obs)
        warm = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        print(f"    [FAIL] infer 抛错: {type(e).__name__}: {e}")
        return 1
    print(f"    第 1 次（含 JIT 编译）: {warm * 1000:.0f} ms")

    times = []
    server_infer_ms = []
    try:
        for _ in range(args.bench_runs):
            t0 = time.perf_counter()
            out = policy.infer(obs)
            times.append(time.perf_counter() - t0)
            # 服务端在响应里回传自己的纯推理耗时，用来把「网络」和「模型」拆开
            if (v := out.get("server_timing", {}).get("infer_ms")) is not None:
                server_infer_ms.append(float(v))
    except Exception as e:  # noqa: BLE001
        print(f"    [FAIL] infer 抛错: {type(e).__name__}: {e}")
        return 1

    arr = np.array(times)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    print(
        f"    客户端侧总计: p50={p50 * 1000:.0f} ms  p95={p95 * 1000:.0f} ms  "
        f"min={arr.min() * 1000:.0f} ms  max={arr.max() * 1000:.0f} ms"
    )
    if server_infer_ms:
        s50 = float(np.percentile(server_infer_ms, 50))
        print(f"    服务端自报推理: p50={s50:.0f} ms")
        print(f"    => 打包 + 网络往返 + 解包 ≈ {p50 * 1000 - s50:.0f} ms")
    if not check_actions(out["actions"], "裸 infer"):
        return 1

    # ---- [4/5] broker 连跑 ----
    print(f"\n[4/5] ActionChunkBroker 连跑 {args.num_steps} 步（action_horizon={args.action_horizon}）")
    broker = action_chunk_broker.ActionChunkBroker(policy=policy, action_horizon=args.action_horizon)
    for i in range(args.num_steps):
        step_obs = robot_env.get_observation() if robot_env is not None else make_offline_obs(i)
        t0 = time.perf_counter()
        action = np.asarray(broker.infer(step_obs)["actions"])
        dt = time.perf_counter() - t0
        if action.shape[-1] != ACTION_DIM:
            print(f"    [FAIL] step {i}: 维度 {action.shape[-1]} != {ACTION_DIM}")
            return 1
        if i < 3 or (i + 1) % 5 == 0:
            print(
                f"    step {i:3d}  xyz=({action[0]:+.4f}, {action[1]:+.4f}, {action[2]:+.4f})  "
                f"grip={action[-1]:+.4f}  ({dt * 1000:.1f} ms)"
            )
    expect_net = -(-args.num_steps // args.action_horizon)  # ceil
    print(
        f"    [ok] {args.num_steps} 步全部返回 {ACTION_DIM} 维动作；"
        f"预期走网络 {expect_net} 次（其余命中 chunk 缓存）"
    )

    # ---- [5/5] 结论 ----
    print("\n[5/5] 结论")
    hz_cap = 1.0 / p95
    print("    链路 OK: 客户端 → websocket → 服务端 → 模型 → 客户端")
    print(f"    单次推理 p50={p50 * 1000:.0f} ms / p95={p95 * 1000:.0f} ms")
    print(f"    按 p95 保守估计，闭环最高约 {hz_cap:.2f} Hz")
    if hz_cap < 7:
        print(f"    [warn] 低于数据集 fps=7 —— 建议 main.py 用 --max-hz {max(1, int(hz_cap))}")
    else:
        print("    [ok] 可以支撑 7 Hz 闭环（数据集 fps=7）")

    if args.use_robot:
        print("    真实观测已通过校验，下一步可以接机器人小步试跑（第 4 步）")
    else:
        print("    建议接着跑一次 --use-robot，验证真实相机图与 TCP→rot6d 链路")

    print("\n全部检查通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
