#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from unittest.mock import Mock

import pytest
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
)
from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_llm.evaluation.decentralized_evaluation_runner import (
    DecentralizedEvaluationRunner,
)
from habitat_llm.utils import setup_config


# A simple dataclass for recording actions, to avoid cross-test imports
@dataclass
class _ActionRec:
    action: str
    params: dict


# Overrides to load the test dataset, from test_planner.py
DATASET_OVERRIDES = [
    "habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/ci.json.gz",
    "habitat.dataset.scenes_dir=data/hssd-partnr-ci",
    "+habitat.dataset.metadata.metadata_folder=data/hssd-partnr-ci/metadata",
    "habitat.environment.iterator_options.shuffle=False",
    "habitat.simulator.agents.agent_1.articulated_agent_urdf=data/humanoids/humanoid_data/female_0/female_0.urdf",
    "habitat.simulator.agents.agent_1.motion_data_path=data/humanoids/humanoid_data/female_0/female_0_motion_data_smplx.pkl",
]


def get_config(config_file, overrides):
    """
    Helper function to load a hydra config.
    This pattern is borrowed from test_planner.py
    """
    with initialize(version_base=None, config_path="../conf"):
        config = compose(
            config_name=config_file,
            overrides=overrides,
        )
    # Manually set up the hydra runtime environment for the test
    HydraConfig().cfg = config
    with open_dict(config):
        config.hydra = {}
        config.hydra.runtime = {}
        config.hydra.runtime.output_dir = "outputs/test"
    return config


def setup_env(config):
    """
    Helper function to set up the environment, from test_planner.py
    """
    register_sensors(config)
    register_actions(config)
    register_measures(config)
    env = EnvironmentInterface(config)
    return env


def test_message_wakes_up_planner(monkeypatch):
    """
    Tests that an agent's planner can be woken up from a 'Done' state
    by a message from another agent. This test adapts the robust setup
    from test_planner.py.
    """
    overrides = [
        "evaluation=decentralized_evaluation_runner_multi_agent",
        "agent@evaluation.agents.agent_0.config=oracle_rearrange_object_states_agent_message",
        "agent@evaluation.agents.agent_1.config=oracle_rearrange_object_states_agent_message",
        "instruct@evaluation.agents.agent_0.planner.plan_config.instruct=zero_shot_prompt",
        "instruct@evaluation.agents.agent_1.planner.plan_config.instruct=zero_shot_prompt",
        "planner@evaluation.agents.agent_0.planner=llm_zero_shot_react_message",
        "planner@evaluation.agents.agent_1.planner=llm_zero_shot_react_message",
        "llm@evaluation.agents.agent_0.planner.plan_config.llm=mock",
        "llm@evaluation.agents.agent_1.planner.plan_config.llm=mock",
        "evaluation.agents.agent_0.planner.plan_config.wake_up=True",
        "evaluation.agents.agent_1.planner.plan_config.wake_up=True",
        "world_model.partial_obs=True",
    ] + DATASET_OVERRIDES

    config = get_config("examples/planner_multi_agent_demo_config.yaml", overrides)

    if not CollaborationDatasetV0.check_config_paths_exist(config.habitat.dataset):
        pytest.skip("Test skipped as dataset files are missing.")

    config = setup_config(config, 0)
    env_interface = setup_env(config)

    # Set up the runner, which will initialize agents and their planners
    eval_runner = DecentralizedEvaluationRunner(config.evaluation, env_interface)

    planner0 = eval_runner.planner[0]
    planner1 = eval_runner.planner[1]

    # Mock the LLM's generate method for planner0
    mock_llm_generate = Mock(return_value='SendMessageTool["time to get back to work"]')
    monkeypatch.setattr(planner0.llm, "generate", mock_llm_generate)

    # 1. Setup Scenario
    # Manually set the planner for Agent 1 to the 'Done' state.
    planner1.is_done = True
    # The SendMessageTool checks this history to see if an agent is done.
    env_interface.agent_action_history[1].append(_ActionRec("Done", {}))

    # 2. Action: Call planner's get_next_action to simulate LLM output processing
    low_level_actions, planner_info, should_end = planner0.get_next_action(
        "test instruction", env_interface.get_observations(), env_interface.world_graph
    )

    # 3. Verification
    # Check that the mock LLM was called
    mock_llm_generate.assert_called_once()

    # Check that the planner's prompt now contains the wake-up message.
    print(planner1.curr_prompt)
    assert "Waking up from Done state" in planner1.curr_prompt
    assert "time to get back to work" in planner1.curr_prompt

    # Check that the planner's state changed
    assert planner1.is_done is False, "Planner for Agent 1 should have been woken up."
    assert (
        planner1.replan_required is True
    ), "Planner should require replan after wake-up."

    # Clean up
    env_interface.env.close()
