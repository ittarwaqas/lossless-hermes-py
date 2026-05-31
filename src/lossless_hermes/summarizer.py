"""LLM summarization with depth-aware prompts for LCM.

Handles structured summarization calls with proper provider integration,
circuit breaker pattern, and depth-specific prompting strategies.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


DEFAULT_LEAF_TARGET_TOKENS = 2400
DEFAULT_CONDENSED_TARGET_TOKENS = 2000

LCM_SUMMARIZER_SYSTEM_PROMPT = (
    "You are a context-compaction summarization engine. Follow user instructions "
    "exactly and return plain text summary content only."
)


@dataclass
class SummaryOptions:
    previous_summary: str | None = None
    is_condensed: bool = False
    depth: int = 0


class LcmProviderAuthError(Exception):
    """Signals that the summarizer hit a provider-auth failure."""

    def __init__(self, provider: str, model: str, message: str):
        super().__init__(message)
        self.provider = provider
        self.model = model


class SummarizerTimeoutError(Exception):
    """Error for summarizer timeouts."""

    pass


class LcmSummarizer:
    """LCM summarization engine with depth-aware prompts."""

    def __init__(
        self,
        provider: str = "",
        model: str = "",
        timeout_ms: int = 60000,
        custom_instructions: str = "",
        circuit_breaker_threshold: int = 5,
        circuit_breaker_cooldown_ms: int = 1800000,
        call_llm_fn: Callable | None = None,
    ):
        self.provider = provider
        self.model = model
        self.timeout_ms = timeout_ms
        self.custom_instructions = custom_instructions
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_cooldown_ms = circuit_breaker_cooldown_ms
        self.call_llm_fn = call_llm_fn

        # Circuit breaker state
        self._failure_count = 0
        self._last_failure_time = 0
        self._circuit_open = False

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        if not self._circuit_open:
            return False

        # Auto-reset after cooldown
        if time.time() * 1000 - self._last_failure_time > self.circuit_breaker_cooldown_ms:
            self._circuit_open = False
            self._failure_count = 0
            return False

        return True

    def _record_failure(self):
        """Record a summarizer failure for circuit breaker."""
        self._failure_count += 1
        self._last_failure_time = time.time() * 1000

        if self._failure_count >= self.circuit_breaker_threshold:
            self._circuit_open = True
            logger.warning(
                f"LCM summarizer circuit breaker opened after {self._failure_count} failures "
                f"(provider: {self.provider}, model: {self.model})"
            )

    def _record_success(self):
        """Record a successful summarizer call."""
        if self._failure_count > 0 or self._circuit_open:
            logger.info(f"LCM summarizer circuit breaker reset (provider: {self.provider}, model: {self.model})")

        self._failure_count = 0
        self._circuit_open = False

    def _build_leaf_prompt(
        self,
        text: str,
        target_tokens: int,
        aggressive: bool = False,
        options: SummaryOptions | None = None,
    ) -> str:
        """Build prompt for leaf-level (depth 0) summarization."""
        options = options or SummaryOptions()

        mode_instruction = "Aggressively compress" if aggressive else "Summarize"

        base_prompt = f"""
{mode_instruction} the following conversation turns into a structured handoff summary.
Preserve key decisions, rationale, constraints, and active tasks. Remove repetition and filler.
Target approximately {target_tokens} tokens.

Your summary MUST end with "Expand for details about: <list of specific topics, file names,
or concepts that were discussed in detail>".

Conversation turns:
{text}

Use this structure:

## What We're Building
[The main goal or project]

## Key Decisions Made
[Important technical decisions and why]

## Current Progress
[What's been completed and what's in progress]

## Active Constraints
[Requirements, preferences, or limitations to remember]

## Files & Tools Used
[Specific files modified, tools used, commands run]

## Next Steps Context
[What needs to happen next - framed as context, not instructions]

## Unresolved Questions
[Any open questions or issues that need attention]

Expand for details about: <specific topics list>"""

        if self.custom_instructions:
            base_prompt += f"\n\nAdditional instructions: {self.custom_instructions}"

        return base_prompt

    def _build_condensed_prompt(
        self,
        text: str,
        target_tokens: int,
        depth: int,
        aggressive: bool = False,
        options: SummaryOptions | None = None,
    ) -> str:
        """Build depth-aware prompt for condensed summarization."""
        options = options or SummaryOptions()

        if depth == 1:
            depth_context = """Compacting leaf-level summaries into condensed memory node.
Focus on timeline progression and major developments. Include hour-level timestamps where relevant."""
        elif depth == 2:
            depth_context = """Condensing session-level summaries.
Emphasize trajectory over minutiae. Include date-level timestamps."""
        else:  # depth >= 3
            depth_context = """Creating high-level memory node.
Focus only on durable context. Use date ranges for timestamps."""

        mode_instruction = "Aggressively compress" if aggressive else "Consolidate"

        base_prompt = f"""
{depth_context}

{mode_instruction} the following summary content into a unified memory node.
Target approximately {target_tokens} tokens.

Content to condense:
{text}

Structure as:

## Session Overview
[What was accomplished overall]

## Key Outcomes
[Durable decisions and results]

## Technical Context
[Architecture, constraints, patterns that matter long-term]

## File & Resource State
[Final state of files, tools, resources]

## Open Items
[Unresolved issues or future work]"""

        if options.previous_summary:
            base_prompt = f"""
Update this existing condensed summary with new information.
PRESERVE all existing relevant context. ADD new developments.
Move completed items appropriately.

EXISTING SUMMARY:
{options.previous_summary}

NEW CONTENT TO INCORPORATE:
{text}

{base_prompt}"""

        if self.custom_instructions:
            base_prompt += f"\n\nAdditional instructions: {self.custom_instructions}"

        return base_prompt

    async def summarize(self, text: str, aggressive: bool = False, options: SummaryOptions | None = None) -> str:
        """Summarize text with appropriate depth-aware prompting."""
        if self._is_circuit_open():
            raise LcmProviderAuthError(self.provider, self.model, "Circuit breaker open - too many recent failures")

        options = options or SummaryOptions()

        # Determine target tokens and prompt type
        if options.is_condensed:
            target_tokens = DEFAULT_CONDENSED_TARGET_TOKENS
            prompt = self._build_condensed_prompt(text, target_tokens, options.depth, aggressive, options)
        else:
            target_tokens = DEFAULT_LEAF_TARGET_TOKENS
            prompt = self._build_leaf_prompt(text, target_tokens, aggressive, options)

        if not self.call_llm_fn:
            raise RuntimeError("No LLM call function provided to summarizer")

        messages = [
            {"role": "system", "content": LCM_SUMMARIZER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            # Use the configured LLM call infrastructure
            response = await self._call_llm_with_timeout(messages, target_tokens * 2)

            content = self._extract_content(response)
            if not content or not content.strip():
                raise RuntimeError("Empty response from summarizer")

            self._record_success()
            return content.strip()

        except Exception as e:
            # Check if this looks like an auth error
            if self._is_auth_error(e):
                self._record_failure()
                raise LcmProviderAuthError(self.provider, self.model, f"Authentication failed: {str(e)}")
            else:
                # Don't trigger circuit breaker for non-auth errors
                logger.warning(f"LCM summarizer call failed: {e}")
                raise

    async def _call_llm_with_timeout(self, messages: list[dict[str, Any]], max_tokens: int):
        """Call LLM with timeout protection."""
        if not self.call_llm_fn:
            raise RuntimeError("No LLM function available")

        call_kwargs = {
            "task": "compression",
            "messages": messages,
            "max_tokens": max_tokens,
            "timeout": self.timeout_ms / 1000.0,  # Convert to seconds
        }

        # Add model overrides if provided
        if self.provider or self.model:
            if self.provider:
                call_kwargs["provider"] = self.provider
            if self.model:
                call_kwargs["model"] = self.model

        return await self.call_llm_fn(**call_kwargs)

    def _extract_content(self, response) -> str:
        """Extract content from LLM response."""
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                content = choice.message.content
                if isinstance(content, str):
                    return content
                elif content is not None:
                    return str(content)

        # Fallback - try to extract from dict format
        if isinstance(response, dict):
            if "choices" in response and response["choices"]:
                choice = response["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    return str(choice["message"]["content"])

        return ""

    def _is_auth_error(self, error: Exception) -> bool:
        """Check if an error looks like an authentication failure."""
        error_text = str(error).lower()

        # Common auth error patterns
        auth_patterns = [
            "401",
            "unauthorized",
            "invalid api key",
            "authentication failed",
            "authorization failed",
            "missing scope",
            "insufficient scope",
            "forbidden",
        ]

        return any(pattern in error_text for pattern in auth_patterns)


# Synchronous wrapper for compatibility
class SyncLcmSummarizer:
    """Synchronous wrapper around LcmSummarizer."""

    def __init__(self, async_summarizer: LcmSummarizer):
        self.async_summarizer = async_summarizer

    def summarize(self, text: str, aggressive: bool = False, options: SummaryOptions | None = None) -> str:
        """Synchronous summarization call."""
        # For now, we'll need to run the async call in a synchronous context
        # This will need to be adapted based on the host's async infrastructure
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we need a different approach
                # This is a placeholder - real implementation depends on the host architecture
                raise RuntimeError("Cannot call async summarizer from async context synchronously")
            else:
                return loop.run_until_complete(self.async_summarizer.summarize(text, aggressive, options))
        except RuntimeError:
            # Create new event loop
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(self.async_summarizer.summarize(text, aggressive, options))
            finally:
                loop.close()
                asyncio.set_event_loop(None)


def create_lcm_summarizer(
    provider: str = "",
    model: str = "",
    timeout_ms: int = 60000,
    custom_instructions: str = "",
    circuit_breaker_threshold: int = 5,
    circuit_breaker_cooldown_ms: int = 1800000,
    call_llm_fn: Callable | None = None,
) -> SyncLcmSummarizer:
    """Create a synchronous LCM summarizer."""
    async_summarizer = LcmSummarizer(
        provider=provider,
        model=model,
        timeout_ms=timeout_ms,
        custom_instructions=custom_instructions,
        circuit_breaker_threshold=circuit_breaker_threshold,
        circuit_breaker_cooldown_ms=circuit_breaker_cooldown_ms,
        call_llm_fn=call_llm_fn,
    )

    return SyncLcmSummarizer(async_summarizer)
