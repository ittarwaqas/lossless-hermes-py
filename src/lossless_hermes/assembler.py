"""Context assembly for LCM.

Reconstructs the context window from the DAG by selecting optimal
summaries and fresh messages within a token budget.
"""

import logging
from dataclasses import dataclass
from typing import Any

from .store.conversation import ConversationStore, MessageRecord
from .store.summary import SummaryRecord, SummaryStore
from .tokens import estimate_tokens

logger = logging.getLogger(__name__)


@dataclass
class AssemblyResult:
    """Result of context assembly."""

    messages: list[dict[str, Any]]
    total_tokens: int
    summary_tokens: int
    message_tokens: int
    summaries_used: int
    messages_used: int
    coverage_ratio: float  # What fraction of history is covered


@dataclass
class AssemblyConfig:
    """Configuration for context assembly."""

    max_tokens: int
    fresh_tail_count: int
    fresh_tail_max_tokens: int | None
    reserve_tokens: int = 1000  # Reserve for response


class ContextAssembler:
    """Assembles context windows from the LCM DAG."""

    def __init__(self, conversation_store: ConversationStore, summary_store: SummaryStore):
        self.conversation_store = conversation_store
        self.summary_store = summary_store

    def assemble_context(self, conversation_id: int, config: AssemblyConfig) -> AssemblyResult:
        """Assemble an optimal context window from the DAG."""

        # Get all available messages
        all_messages = self.conversation_store.get_messages_by_conversation(conversation_id)
        if not all_messages:
            return AssemblyResult(
                messages=[],
                total_tokens=0,
                summary_tokens=0,
                message_tokens=0,
                summaries_used=0,
                messages_used=0,
                coverage_ratio=0.0,
            )

        # Calculate available budget (reserve some for response)
        available_tokens = config.max_tokens - config.reserve_tokens

        # Step 1: Protect fresh tail
        fresh_tail = self._select_fresh_tail(all_messages, config)
        fresh_tail_tokens = sum(estimate_tokens(self._message_to_text(msg)) for msg in fresh_tail)

        # Step 2: Budget remaining after fresh tail
        remaining_budget = available_tokens - fresh_tail_tokens

        if remaining_budget <= 0:
            # Fresh tail alone exceeds budget
            truncated_tail = self._truncate_messages_to_budget(fresh_tail, available_tokens)
            return AssemblyResult(
                messages=[self._message_to_dict(msg) for msg in truncated_tail],
                total_tokens=sum(estimate_tokens(self._message_to_text(msg)) for msg in truncated_tail),
                summary_tokens=0,
                message_tokens=sum(estimate_tokens(self._message_to_text(msg)) for msg in truncated_tail),
                summaries_used=0,
                messages_used=len(truncated_tail),
                coverage_ratio=len(truncated_tail) / len(all_messages),
            )

        # Step 3: Select optimal summaries for remaining budget
        optimal_summaries = self._select_optimal_summaries(conversation_id, remaining_budget, fresh_tail)

        # Step 4: Assemble final context
        context_messages = []

        # Add summaries first (chronological order)
        for summary in optimal_summaries:
            summary_message = self._summary_to_message(summary)
            context_messages.append(summary_message)

        # Add fresh tail
        for message in fresh_tail:
            context_messages.append(self._message_to_dict(message))

        # Calculate final statistics
        summary_tokens = sum(estimate_tokens(summary.content) for summary in optimal_summaries)
        message_tokens = fresh_tail_tokens
        total_tokens = summary_tokens + message_tokens

        # Estimate coverage
        covered_messages = len(fresh_tail)
        for summary in optimal_summaries:
            covered_messages += summary.descendant_count

        coverage_ratio = min(1.0, covered_messages / len(all_messages))

        return AssemblyResult(
            messages=context_messages,
            total_tokens=total_tokens,
            summary_tokens=summary_tokens,
            message_tokens=message_tokens,
            summaries_used=len(optimal_summaries),
            messages_used=len(fresh_tail),
            coverage_ratio=coverage_ratio,
        )

    def _select_fresh_tail(self, messages: list[MessageRecord], config: AssemblyConfig) -> list[MessageRecord]:
        """Select fresh tail messages to preserve."""
        if not messages:
            return []

        # Start with configured count
        protected_count = min(config.fresh_tail_count, len(messages))

        # Apply token budget if configured
        if config.fresh_tail_max_tokens:
            token_budget = config.fresh_tail_max_tokens
            accumulated_tokens = 0

            # Walk backwards from the end
            for i in range(len(messages) - 1, -1, -1):
                msg_tokens = estimate_tokens(self._message_to_text(messages[i]))
                if accumulated_tokens + msg_tokens > token_budget:
                    break
                accumulated_tokens += msg_tokens

            # Use the more restrictive constraint
            token_protected = len(messages) - i - 1
            protected_count = min(protected_count, token_protected)

        # Always protect at least the last message
        protected_count = max(1, protected_count)

        return messages[-protected_count:]

    def _select_optimal_summaries(
        self, conversation_id: int, budget: int, fresh_tail: list[MessageRecord]
    ) -> list[SummaryRecord]:
        """Select optimal summaries to fit within the budget."""

        # Get all summaries for this conversation
        all_summaries = self.summary_store.get_summaries_by_conversation(conversation_id)
        if not all_summaries:
            return []

        # Exclude summaries that overlap with fresh tail
        # For simplicity, we'll just use temporal boundaries
        if fresh_tail:
            earliest_fresh = min(msg.created_at for msg in fresh_tail)
            candidate_summaries = [s for s in all_summaries if s.latest_at is None or s.latest_at < earliest_fresh]
        else:
            candidate_summaries = all_summaries

        if not candidate_summaries:
            return []

        # Sort by efficiency (coverage per token)
        def summary_efficiency(summary: SummaryRecord) -> float:
            if summary.token_count <= 0:
                return 0.0
            return summary.descendant_token_count / summary.token_count

        candidate_summaries.sort(key=summary_efficiency, reverse=True)

        # Greedy selection by efficiency
        selected = []
        used_budget = 0

        for summary in candidate_summaries:
            if used_budget + summary.token_count <= budget:
                selected.append(summary)
                used_budget += summary.token_count

        # Sort selected summaries chronologically
        selected.sort(key=lambda s: s.earliest_at or s.created_at)

        return selected

    def _truncate_messages_to_budget(self, messages: list[MessageRecord], budget: int) -> list[MessageRecord]:
        """Truncate messages to fit within token budget."""
        if not messages:
            return []

        # Always keep at least the last message
        if len(messages) == 1:
            return messages

        # Walk backwards from the end until we exceed budget
        accumulated_tokens = 0

        for i in range(len(messages) - 1, -1, -1):
            msg_tokens = estimate_tokens(self._message_to_text(messages[i]))
            if accumulated_tokens + msg_tokens > budget and i < len(messages) - 1:
                break
            accumulated_tokens += msg_tokens

        return messages[i + 1 :]

    def _message_to_text(self, message: MessageRecord) -> str:
        """Convert message to text for token estimation."""
        return f"[{message.role.upper()}]: {message.content}"

    def _message_to_dict(self, message: MessageRecord) -> dict[str, Any]:
        """Convert MessageRecord to OpenAI message dict.

        Preserves tool_call linkage: tool-role messages include tool_call_id
        so MiniMax can match tool results back to the originating tool_calls.
        """
        result: dict[str, Any] = {"role": message.role, "content": message.content}

        # For tool messages, include the tool_call_id from message_parts so
        # MiniMax can link the result back to the assistant's tool_call.
        if message.role == "tool":
            parts = self.conversation_store.get_message_parts(message.message_id)
            for part in parts:
                if part.part_type == "tool" and part.tool_call_id:
                    result["tool_call_id"] = part.tool_call_id
                    break

        # For assistant messages that have tool_calls, reconstruct them from
        # message_parts so the API can issue tool_call results.
        if message.role == "assistant":
            parts = self.conversation_store.get_message_parts(message.message_id)
            tool_calls = []
            for part in parts:
                if part.part_type == "tool" and part.tool_call_id and part.tool_name:
                    import json as _json
                    tool_input = {}
                    if part.tool_input:
                        try:
                            tool_input = _json.loads(part.tool_input)
                        except Exception:
                            tool_input = {"raw": part.tool_input}
                    tool_calls.append({
                        "id": part.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": part.tool_name or "",
                            "arguments": _json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input),
                        },
                    })
            if tool_calls:
                result["tool_calls"] = tool_calls

        return result

    def _summary_to_message(self, summary: SummaryRecord) -> dict[str, Any]:
        """Convert SummaryRecord to OpenAI message dict."""
        # Summaries are injected as assistant messages with special markers
        content = f"[LCM CONTEXT SUMMARY - Depth {summary.depth}]\n\n{summary.content}"

        return {"role": "assistant", "content": content}

    def save_context_assembly(self, conversation_id: int, result: AssemblyResult):
        """Save the assembled context state for debugging/analysis."""

        # Clear existing context items
        self.summary_store.clear_context_items(conversation_id)

        position = 0

        # Record which summaries and messages were used
        for message_dict in result.messages:
            content = message_dict.get("content", "")

            if content.startswith("[LCM CONTEXT SUMMARY"):
                # This is a summary - we'd need to track which one
                # For now, just record as summary type
                self.summary_store.add_context_item(
                    conversation_id=conversation_id,
                    position=position,
                    item_type="summary",
                    summary_id=None,  # Would need better tracking
                )
            else:
                # This is a regular message
                # We'd need to match it back to a message_id
                self.summary_store.add_context_item(
                    conversation_id=conversation_id,
                    position=position,
                    item_type="message",
                    message_id=None,  # Would need better tracking
                )

            position += 1
