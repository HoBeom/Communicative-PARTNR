# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

from typing import TYPE_CHECKING, Dict

from habitat_llm.llm.instruct.utils import get_objects_descr
from habitat_llm.planner import LLMPlanner
from habitat_llm.utils.grammar import FREE_TEXT

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from habitat_llm.agent.env import EnvironmentInterface
    from habitat_llm.world_model.world_graph import WorldGraph


class ZeroShotReactMessagePlanner(LLMPlanner):
    """
    This class builds the prompt for the, zero shot llm react planner format.
    """

    def __init__(
        self, plan_config: "DictConfig", env_interface: "EnvironmentInterface"
    ) -> None:
        """
        Initialize the ZeroShotReactPlanner.

        :param plan_config: The planner configuration.
        :param env_interface: The environment interface.
        """
        super().__init__(plan_config, env_interface)
        self.wake_up = self.planner_config.get("wake_up", False)
        # print(self.agents)
        # # assert len(self.agents) == 1, "ZeroShotReactMessagePlanner only supports decentralized planning"
        # self.agent_uid = self.agents[0].uid
        # self.env_interface.set_planner(self.agent_uid, self)
        # print(f"{self.agents[0].uid} wake_up: {self.wake_up}")

    def notify_agent_wakeup(self) -> bool:
        """
        Attempt to wake up the agent in the planner. Only applies if wake_up is enabled
        and the planner is currently in a 'Done' state.

        Returns:
            bool: True if the agent is successfully woken up (state changed).
        """
        if not self.wake_up:
            return False

        if not self.is_done:
            return False

        # Mark planner as active again
        self.is_done = False
        self.replan_required = True

        # Clear old execution traces
        agent_uid = self.agents[0].uid
        self.latest_agent_response.pop(agent_uid, None)
        self.last_high_level_actions.pop(agent_uid, None)

        # Log wake-up to prompt and trace
        user_tag = self.planner_config.llm.user_tag
        assistant_tag = self.planner_config.llm.assistant_tag
        eot_tag = self.planner_config.llm.eot_tag

        new_messages = self.env_interface.get_new_messages(agent_uid)
        messages_text = "\n".join(
            [
                f"Message from Agent {msg['sender']}: {msg['message']}"
                for msg in new_messages
            ]
        )
        prompt = f"{user_tag}Received a message. Waking up from Done state.\n{messages_text}\n{eot_tag}"
        self.curr_prompt += prompt + assistant_tag
        self.trace += prompt + assistant_tag
        print(prompt + assistant_tag, end="")

        return True

    def build_response_grammar(self, world_graph: "WorldGraph") -> str:
        """
        Build a grammar that accepts all valid responses based on a world graph.

        :param world_graph: The world graph.
        :return: The response grammar.
        """
        delimiter = "\\n"
        tool_rules = self.build_tool_grammar(world_graph)

        root_rule = (
            f'root ::= {FREE_TEXT} "{delimiter}" tool_call "{delimiter}Assigned!"'
        )

        return "\n".join([root_rule, tool_rules])

    def _add_responses_to_prompt(self, responses: Dict[int, str]) -> str:
        """
        Add skill responses to the prompt optionally including object descriptions and new messages.
        The new messages field includes only the newly received messages.

        :param responses: A dictionary of agent responses.
        :return: The updated print string.
        """
        if self.planner_config.objects_response:
            assert len(self.agents) == 1
            agent = self.agents[0]
            # print(f"{agent.uid=}, {self.agents=}")
            result = ""
            world_graph = self.env_interface.world_graph[agent.uid]
            if responses[agent.uid] != "":
                objects = get_objects_descr(
                    world_graph,
                    agent.uid,
                    include_room_name=True,
                    add_state_info=self.planner_config.objects_response_include_states,
                )
                new_messages = self.env_interface.get_new_messages(agent.uid)
                if new_messages:
                    messages_text = "\n".join(
                        [
                            f"Message from Agent {msg['sender']}: {msg['message']}"
                            for msg in new_messages
                        ]
                    )
                    response_format = "{user_tag}Result: {result}\nObjects: {objects}\n{messages}{eot_tag}"
                    result = response_format.format(
                        result=responses[agent.uid],
                        objects=objects,
                        messages=messages_text,
                        user_tag=self.planner_config.llm.user_tag,
                        eot_tag=self.planner_config.llm.eot_tag,
                    )
                else:
                    response_format = (
                        "{user_tag}Result: {result}\nObjects: {objects}\n{eot_tag}"
                    )
                    result = response_format.format(
                        result=responses[agent.uid],
                        objects=objects,
                        user_tag=self.planner_config.llm.user_tag,
                        eot_tag=self.planner_config.llm.eot_tag,
                    )
                self.curr_prompt += result + self.planner_config.llm.assistant_tag
                print(result + self.planner_config.llm.assistant_tag, end="")
                self.trace += result + self.planner_config.llm.assistant_tag
        else:
            result = super()._add_responses_to_prompt(responses)
        return result
