"""Inspect AI entrypoint for tasks and hooks."""

# Import side effects register tasks and hooks for Inspect discovery.
from he_rel.tasks.humanevalrel import humanevalrel

from he_rel.scoring.code_tester import verify
from he_rel.utils.code_extraction import extract_markdown_block

default_humaneval_scorer = verify( extract_markdown_block )

from he_rel.solvers.chain_of_prompts import chain_of_prompts

########################
# MonkeyPatch OpenRouterAPI to support extra_body and extra_headers this is useful from some models that support max_reasoning token settings.
from inspect_ai.model._providers.openrouter import OpenRouterAPI  # noqa: E402
from typing import Any  # noqa: E402
from inspect_ai.model import GenerateConfig  # noqa: E402
from openai.types.chat import ChatCompletion  # noqa: E402

_original_init = OpenRouterAPI.__init__
_original_generate_completion = OpenRouterAPI._generate_completion

def _new_init(self, *args, **kwargs):
    self._extra_body = kwargs.pop("extra_body", None)
    self._extra_headers = kwargs.pop("extra_headers", None)
    _original_init(self, *args, **kwargs)

async def _new_generate_completion(self, request: dict[str, Any], config: GenerateConfig) -> ChatCompletion:
    if getattr(self, "_extra_body", None):
        print("Adding extra_body to OpenRouter request:", self._extra_body)
        request["extra_body"] = {**request.get("extra_body", {}), **self._extra_body}
    
    if getattr(self, "_extra_headers", None):
        request["extra_headers"] = {**request.get("extra_headers", {}), **self._extra_headers}
        
    return await _original_generate_completion(self, request, config)

OpenRouterAPI.__init__ = _new_init  # ty:ignore[invalid-assignment]
OpenRouterAPI._generate_completion = _new_generate_completion  # ty:ignore[invalid-assignment]
########################


__all__ = ["humanevalrel", "default_humaneval_scorer", "chain_of_prompts"]