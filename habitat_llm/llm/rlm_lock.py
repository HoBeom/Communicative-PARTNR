#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import filelock

# from rlm.llm import RemoteLanguageModel
import requests
from omegaconf import DictConfig


class RemoteLanguageModel:
    """
    RemoteLanguageModel communicates with a remote vLLM server to generate text.
    """

    def __init__(self, conf: DictConfig, chat: bool = False):
        self.llm_conf = conf
        host, port = self.llm_conf.host, self.llm_conf.port
        self.generation_params = self.llm_conf.generation_params
        self.model_name = self.generation_params.engine
        self.api_address = f"http://{host}:{port}"
        self.chat = chat

    def generate(
        self,
        prompt: str,
        stop: Optional[str] = None,
        max_new_tokens: Optional[int] = 250,
        temperature: Optional[float] = 0.0,  # ignore temperature
        sampling: Optional[bool] = False,  # ignore sampling
        generation_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generates text for a single prompt by calling batch_generate() and returning the first result.

        :param prompt: The input prompt (already containing all necessary tags).
        :param max_new_tokens: The maximum number of tokens to generate.
        :param temperature: Temperature for generation.
        :param sampling: Whether to use sampling.
        :param generation_args: Additional generation parameters to include in the payload.
        :return: A dictionary with the generated text (e.g., {"generation": "generated text"}).
        """
        if self.chat:
            raise NotImplementedError("Chat not implemented yet")
            return self.batch_chat_generate(
                [prompt], stop, max_new_tokens, generation_args=generation_args
            )[0]
        return self.batch_generate(
            [prompt], max_new_tokens, generation_args=generation_args  # type: ignore[arg-type]
        )[0]

    def batch_generate(
        self,
        prompts: List[str],
        stop: Optional[str] = None,
        max_length: Optional[int] = 250,
        generation_args: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generates text for multiple prompts by sending a request to the remote LLM server.
        Extra parameters in generation_args are merged into the payload.

        :param
        prompts: A list of input prompts (each already formatted with necessary tags).
        :param max_length: Maximum number of tokens to generate.
        :param temperature: Temperature for generation.
        :param sampling: Whether to use sampling.
        :param generation_args: Additional generation parameters (e.g., repetition_penalty, top_k, etc.).
        :return: A list of dictionaries, each containing a "generation" field with the generated text.
        """

        headers = {"Content-Type": "application/json"}
        results = []
        url = f"{self.api_address}/v1/completions"
        try:
            for prompt in prompts:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "max_tokens": max_length,
                    "temperature": 0.0,
                    "n": 1,
                }
                # if generation_args:
                #     payload.update(generation_args)
                response = requests.post(url, headers=headers, json=payload, timeout=40)
                response.raise_for_status()
                result = response.json()
                generated_text = result["choices"][0]["text"]
                results.append({"generation": generated_text})
        except Exception as e:
            results.append({"generation": f"Error calling remote model: {e}"})

        return results

    def batch_chat_generate(
        self,
        prompts: List[str],
        stop: Optional[str] = None,
        max_length: Optional[int] = 250,
        generation_args: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generates text for multiple prompts by sending a request to the remote LLM server.
        Extra parameters in generation_args are merged into the payload.

        :param prompts: A list of input prompts (each already formatted with necessary tags).
        :param max_length: Maximum number of tokens to generate.
        :param temperature: Temperature for generation.
        :param sampling: Whether to use sampling.
        :param generation_args: Additional generation parameters (e.g., repetition_penalty, top_k, etc.).
        :return: A list of dictionaries, each containing a "generation" field with the generated text.
        """
        headers = {"Content-Type": "application/json"}
        results = []
        url = f"{self.api_address}/v1/chat/completions"
        try:
            for prompt in prompts:
                payload = {
                    "model": self.model_name,
                    # TODO system prompt should be separate.
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_length,
                    "temperature": 0.0,
                    "n": 1,
                }
                if generation_args:
                    payload.update(generation_args)
                response = requests.post(url, headers=headers, json=payload, timeout=40)
                response.raise_for_status()
                result = response.json()
                generated_text = result["choices"][0]["message"]["content"]
                results.append({"generation": generated_text})
        except Exception as e:
            results.append({"generation": f"Error calling remote model: {e}"})

        return results


class RemotePoolLanguageModel(RemoteLanguageModel):
    """
    Allows multiple processes to access a pool of llms. Works by keeping a json of
    LLM addresses and the number of agents that are on the queue for each address
    when called, assigns an agent with the llm with the shortest queue.
    In particular let's say that the serve_model function initialized
    4 nodes, with the files inside out_dir/{hostname_i}:{port_i}.
    This LanguageModel will create a JSON of the form
    {
      "{hostname_i}:{port_i}": ct_i
    }
    for i 1..4. Where ct_i will measure how many processes are calling
    an llm in that host. This call keeps track of this JSON and whenever a new
    process wants to call an LLM, it assigns the port with lower ct_i
    """

    def __init__(self, serverdir: str) -> None:
        self.addresses = [
            f"http://{p.name}" for p in Path(f"{serverdir}/server_list/").glob("*")
        ]
        self.lockfile = f"{serverdir}/lock.json"
        self.lockfilelock = f"{serverdir}/lock.json.lock"

    def get_address_with_shortest_queue(self):
        """
        Get the address of the LLM that has less processes waiting,
        and increase the number of processes waiting for that LLM
        """
        lockfile = self.lockfile
        lockfilelock = self.lockfilelock
        t_address = None
        with filelock.FileLock(lockfilelock):
            with open(lockfile, "r") as f:
                content = json.load(f)
            # Sort addresses by number of calls, and return the first one
            # this is to get the address with lowest number of calls.
            t_address = sorted(content.items(), key=lambda kv: kv[1])[0][0]
            content[t_address] += 1
            with open(lockfile, "w+") as f:
                f.write(json.dumps(content))
        return t_address

    def free_address(self, address):
        """
        Whenever the LLM completes the call reduce the number of processes
        pointing to that LLM
        """
        lockfile = self.lockfile
        lockfilelock = self.lockfilelock
        with filelock.FileLock(lockfilelock):
            with open(lockfile, "r") as f:
                content = json.load(f)
            content[address] -= 1
            with open(lockfile, "w+") as f:
                f.write(json.dumps(content))

    def batch_generate(  # type: ignore[override]
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float = 1.0,
        sampling: bool = False,
        generation_args: Dict = None,
    ) -> List[Dict]:
        """
        Generate an LLM output. Find the most free LLM, assign the process to that LLM
        when done, free that LLM.
        """
        lockfile = self.lockfile
        lockfilelock = self.lockfilelock
        # If the json file does not exist, start it here
        with filelock.FileLock(lockfilelock):
            if not os.path.isfile(lockfile):
                with open(lockfile, "w+") as f:
                    lock_contents = {addr: 0 for addr in self.addresses}
                    f.write(json.dumps(lock_contents))
        # Get the llm with fewer calls
        self.address = self.get_address_with_shortest_queue()
        # Run the llm with te given prompt
        result = super().batch_generate(  # type: ignore[misc]
            prompts,
            max_new_tokens,  # type: ignore[arg-type]
            temperature,  # type: ignore[arg-type]
            sampling,  # type: ignore[arg-type]
            generation_args=generation_args,
        )
        # Free that llm in the json file.
        self.free_address(self.address)
        return result
