#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Dict, List, Tuple
from unittest.mock import Mock

import pytest

from habitat_llm.tools.perception.send_message_tool import SendMessageTool

# -----------------------------
# Helpers & lightweight fakes
# -----------------------------
SentRecord = Tuple[int, int, str]  # (sender_id, receiver_id, message)


@dataclass
class _ActionRec:
    action: str  # e.g., "Navigate[table_1]" or "... Done"


class _FakeEnv:
    def __init__(
        self,
        agent_action_history: Dict[int, List[_ActionRec]],
        can_wake_map: Dict[int, bool] = None,
    ):
        self.agent_action_history = agent_action_history
        self._sent: List[SentRecord] = []
        self._woken: List[int] = []
        self._can_wake = can_wake_map or {}

    # API consumed by SendMessageTool
    def send_message_to_agent(self, sender_id, receiver_id, message):
        self._sent.append((sender_id, receiver_id, message))

    def can_wake_up_agent(self, uid):
        return self._can_wake.get(uid, False)

    def wake_up_agent(self, uid):
        self._woken.append(uid)


class _SkillCfg:
    """
    Minimal config carrying name/prompt/description and dict-like get().
    Also allows attribute access used in __init__ (e.g., .name).
    """

    def __init__(
        self,
        name="send_message",
        prompt="Message_Gen",
        description="Send messages among agents",
        system_feedback_mode="Opaque",
        speech_recognition_range=7.0,
    ):
        self.name = name
        self.prompt = prompt
        self.description = description
        self._store = {
            "system_feedback_mode": system_feedback_mode,
            "speech_recognition_range": speech_recognition_range,
        }

    def get(self, key, default=None):
        return self._store.get(key, default)


@pytest.fixture
def tool_opaque():
    return SendMessageTool(_SkillCfg(system_feedback_mode="Opaque"))


@pytest.fixture
def tool_binary():
    return SendMessageTool(_SkillCfg(system_feedback_mode="Binary"))


@pytest.fixture
def tool_causal():
    return SendMessageTool(_SkillCfg(system_feedback_mode="Causal"))


@pytest.fixture
def tool_traceable():
    return SendMessageTool(_SkillCfg(system_feedback_mode="Traceable"))


# -----------------------------
# Basic behavior
# -----------------------------
def test_initialization_name_and_description(tool_binary):
    assert tool_binary.name == "send_message"
    assert isinstance(tool_binary.description, str)
    assert tool_binary.argument_types == ["message_topic_or_content"]


def test_set_environment_and_optional_llm(tool_binary):
    env = _FakeEnv({0: [_ActionRec("Idle")], 1: [_ActionRec("Idle")]})
    tool_binary.set_environment(env)
    assert tool_binary.env_interface is env

    # LLM is optional for current implementation; ensure set_llm doesn't crash
    mock_llm = Mock()
    mock_llm.llm_conf = {
        "system_tag": "SYSTEM",
        "user_tag": "USER",
        "assistant_tag": "ASSISTANT",
        "eot_tag": "EOT",
    }
    tool_binary.set_llm(mock_llm)
    assert tool_binary.llm is mock_llm


def test_missing_sender_id_raises(tool_binary):
    env = _FakeEnv({0: [_ActionRec("Idle")], 1: [_ActionRec("Idle")]})
    tool_binary.set_environment(env)
    with pytest.raises(ValueError):
        tool_binary.process_action("hello", observations={})  # no agent_id


def test_missing_environment_raises(tool_binary):
    with pytest.raises(ValueError):
        tool_binary.process_action("hello", observations={"agent_id": 0})


def test_empty_message_reports_failure_causal(tool_causal, monkeypatch):
    env = _FakeEnv({0: [_ActionRec("Idle")], 1: [_ActionRec("Idle")]})
    tool_causal.set_environment(env)

    # proximity result doesn't matter; message is empty
    monkeypatch.setattr(
        tool_causal,
        "_is_agent_in_proximity",
        lambda s, r: (True, (3.0, "roomA", "roomA")),
    )

    _, msg = tool_causal.process_action("   ", {"agent_id": 0})
    assert "empty message field" in msg


# -----------------------------
# Proximity & delivery outcomes
# -----------------------------
def test_success_delivery_binary(tool_binary, monkeypatch):
    env = _FakeEnv({0: [_ActionRec("Idle")], 1: [_ActionRec("Idle")]})
    tool_binary.set_environment(env)

    # Force proximity True (same room), distance returned for message text assembly
    monkeypatch.setattr(
        tool_binary,
        "_is_agent_in_proximity",
        lambda s, r: (True, (2.5, "roomX", "roomX")),
    )

    _, msg = tool_binary.process_action("hello", {"agent_id": 0})
    assert "Successfully sent message" in msg
    assert "Agent 1" in msg  # Binary mode lists successful recipients
    assert env._sent == [(0, 1, "hello")]


def test_success_delivery_opaque_unsure(tool_opaque, monkeypatch):
    env = _FakeEnv({0: [_ActionRec("Idle")], 1: [_ActionRec("Idle")]})
    tool_opaque.set_environment(env)

    monkeypatch.setattr(
        tool_opaque, "_is_agent_in_proximity", lambda s, r: (True, (1.0, "A", "A"))
    )

    _, msg = tool_opaque.process_action("ping", {"agent_id": 0})
    assert "unclear if the other agent received" in msg
    # Message is actually sent, but response communicates uncertainty
    assert env._sent == [(0, 1, "ping")]


def test_failure_no_proximity_causal(tool_causal, monkeypatch):
    env = _FakeEnv({0: [_ActionRec("Idle")], 1: [_ActionRec("Idle")]})
    tool_causal.set_environment(env)

    # Force proximity False; causal mode includes reason but not per-agent trace
    monkeypatch.setattr(
        tool_causal,
        "_is_agent_in_proximity",
        lambda s, r: (False, (12.3, "roomA", "roomB")),
    )

    _, msg = tool_causal.process_action("hello", {"agent_id": 0})
    assert msg.startswith("Message send failed")
    assert "within 7.0 meters" in msg
    # Causal mode does not include per-agent trace like room/distance lines.


def test_failure_trace_includes_done_state(tool_traceable, monkeypatch):
    # Receiver is Done and cannot be woken — trace shows Done
    env = _FakeEnv(
        {
            0: [_ActionRec("Idle")],
            1: [_ActionRec("Navigate[table]"), _ActionRec("Done")],
        },
        can_wake_map={1: False},
    )
    tool_traceable.set_environment(env)
    monkeypatch.setattr(
        tool_traceable,
        "_is_agent_in_proximity",
        lambda s, r: (False, (9.9, "R1", "R2")),
    )

    _, msg = tool_traceable.process_action("hello", {"agent_id": 0})
    assert "[Agent 1: Done]" in msg


def test_wake_up_path_when_done_and_can_wake(tool_traceable, monkeypatch):
    # Receiver is Done but can be woken — send + wake_up called
    env = _FakeEnv(
        {
            0: [_ActionRec("Idle")],
            1: [_ActionRec("Done")],
        },
        can_wake_map={1: True},
    )
    tool_traceable.set_environment(env)
    monkeypatch.setattr(
        tool_traceable,
        "_is_agent_in_proximity",
        lambda s, r: (True, (4.2, "R", "R")),
    )

    _, msg = tool_traceable.process_action("hello", {"agent_id": 0})
    assert "Successfully sent message" in msg
    assert env._sent == [(0, 1, "hello")]
    assert env._woken == [1]


# -----------------------------
# Validation of config values
# -----------------------------
@pytest.mark.parametrize(
    "bad_value, expected_text",
    [
        (-1, "should be positive"),
        ("not-a-number", "should be a number"),
    ],
)
def test_invalid_speech_recognition_range_raises(bad_value, expected_text):
    with pytest.raises(ValueError) as ei:
        SendMessageTool(_SkillCfg(speech_recognition_range=bad_value))
    assert expected_text in str(ei.value)
