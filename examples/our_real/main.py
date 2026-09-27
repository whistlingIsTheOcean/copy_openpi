import dataclasses
import logging

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.our_real import env as _env


@dataclasses.dataclass
class Args:
    
    host: str
    port: int = 8001 

    action_horizon: int = 25

    num_episodes: int = 1
    max_episode_steps: int = 1000

    
    max_hz: float = 7.0

    # False = dry-run：读真实观测、连服务端、打印动作，但一个字节都不发给机器人
    send_actions: bool = True


def main(args: Args) -> None:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")
    logging.info(f"Connecting to ws://{args.host}:{args.port}, max_hz={args.max_hz}")

    runtime = _runtime.Runtime(
        environment=_env.FlexivRealEnvironment(
            discretize_rotation=False,
            send_actions=args.send_actions,
        ),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[],
        max_hz=args.max_hz,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
