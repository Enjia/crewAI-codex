from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from crewai.llms.base_llm import BaseLLM

if TYPE_CHECKING:
    from crewai.task import Task
    from crewai.agent.core import Agent
    from crewai.utilities.types import LLMMessage


class CodexCLICompletion(BaseLLM):
    """Codex CLI-based completion provider.

    This provider shells out to the `codex` CLI using `codex exec` and captures
    the final assistant message via `--output-last-message`.

    Notes:
        - This does not support CrewAI tool calling. Any provided tools are ignored.
        - Set Codex config via `codex_config_overrides` or environment (~/.codex/config.toml).
        - You can pass a JSON schema via `response_model` and Codex will try to comply.
    """

    def __init__(
        self,
        model: str = "gpt-5.2-codex",
        *,
        codex_path: str | None = None,
        codex_config_overrides: dict[str, Any] | list[str] | None = None,
        codex_cd: str | None = None,
        codex_add_dir: list[str] | None = None,
        codex_skip_git_repo_check: bool = True,
        codex_sandbox: str | None = None,
        codex_full_auto: bool = False,
        codex_dangerously_bypass_approvals_and_sandbox: bool = False,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        provider = kwargs.pop("provider", "codex_cli")
        super().__init__(model=model, timeout=timeout, provider=provider, **kwargs)

        self.codex_path = codex_path or "codex"
        if shutil.which(self.codex_path) is None:
            raise FileNotFoundError(
                f"Codex CLI not found: {self.codex_path}. Install it and ensure it's on PATH."
            )

        self.codex_config_overrides = codex_config_overrides
        self.codex_cd = codex_cd
        self.codex_add_dir = codex_add_dir or []
        self.codex_skip_git_repo_check = codex_skip_git_repo_check
        self.codex_sandbox = codex_sandbox
        self.codex_full_auto = codex_full_auto
        self.codex_dangerously_bypass_approvals_and_sandbox = (
            codex_dangerously_bypass_approvals_and_sandbox
        )
        self.timeout = timeout

    def call(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str:
        prompt = _messages_to_prompt(messages)

        # Build codex exec command
        cmd = [self.codex_path, "exec", "-m", self.model]

        if self.codex_cd:
            cmd += ["--cd", self.codex_cd]
        if self.codex_skip_git_repo_check:
            cmd.append("--skip-git-repo-check")
        if self.codex_sandbox:
            cmd += ["--sandbox", self.codex_sandbox]
        if self.codex_full_auto:
            cmd.append("--full-auto")
        if self.codex_dangerously_bypass_approvals_and_sandbox:
            cmd.append("--dangerously-bypass-approvals-and-sandbox")
        for extra_dir in self.codex_add_dir:
            cmd += ["--add-dir", extra_dir]

        # Apply config overrides
        cmd += _format_codex_overrides(self.codex_config_overrides)

        # Response schema (optional)
        schema_file = None
        if response_model is not None:
            schema_fd, schema_path = tempfile.mkstemp(suffix=".json")
            schema_file = schema_path
            with os.fdopen(schema_fd, "w") as f:
                json.dump(response_model.model_json_schema(), f)
            cmd += ["--output-schema", schema_file]

        # Capture final response into a temp file
        output_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False) as out_f:
                output_path = out_f.name
            cmd += ["--output-last-message", output_path]

            result = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=self.timeout,
            )

            if result.returncode != 0:
                stderr = (result.stderr or "").strip()
                stdout = (result.stdout or "").strip()
                details = stderr or stdout or "codex exec failed"
                raise RuntimeError(details)

            if output_path and os.path.exists(output_path):
                with open(output_path, "r", encoding="utf-8") as f:
                    return f.read().strip()

            # Fallback to stdout if output file missing
            return (result.stdout or "").strip()
        finally:
            if output_path and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            if schema_file and os.path.exists(schema_file):
                try:
                    os.remove(schema_file)
                except OSError:
                    pass


def _format_codex_overrides(
    overrides: dict[str, Any] | list[str] | None,
) -> list[str]:
    if not overrides:
        return []
    if isinstance(overrides, list):
        return [item for item in overrides if isinstance(item, str)]
    args: list[str] = []
    for key, value in overrides.items():
        if value is None:
            continue
        # Let codex parse TOML; encode basic values as JSON for safety.
        try:
            value_str = json.dumps(value)
        except TypeError:
            value_str = str(value)
        args += ["-c", f"{key}={value_str}"]
    return args


def _messages_to_prompt(messages: str | list[LLMMessage]) -> str:
    if isinstance(messages, str):
        return messages

    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Flatten multimodal content into text
            content = " ".join(str(part) for part in content)
        parts.append(f"{role.upper()}: {content}")
    return "\n".join(parts).strip()
