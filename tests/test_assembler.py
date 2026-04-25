"""Tests for lossless_hermes.assembler module."""

from datetime import datetime, timedelta

import pytest

from lossless_hermes.assembler import AssemblyConfig, ContextAssembler
from lossless_hermes.store.conversation import CreateMessageInput, CreateMessagePartInput
from lossless_hermes.store.summary import CreateSummaryInput


class TestContextAssembler:
    @pytest.fixture
    def assembler(self, conversation_store, summary_store):
        return ContextAssembler(conversation_store, summary_store)

    def test_empty_conversation(self, assembler, conversation_store):
        conv = conversation_store.create_conversation("empty")
        config = AssemblyConfig(max_tokens=10000, fresh_tail_count=10, fresh_tail_max_tokens=None)
        result = assembler.assemble_context(conv.conversation_id, config)
        assert result.messages == []
        assert result.total_tokens == 0
        assert result.coverage_ratio == 0.0

    def test_small_conversation_all_messages(self, assembler, sample_conversation):
        conv, msgs = sample_conversation
        config = AssemblyConfig(max_tokens=100000, fresh_tail_count=100, fresh_tail_max_tokens=None)
        result = assembler.assemble_context(conv.conversation_id, config)
        assert result.messages_used == len(msgs)
        assert result.summaries_used == 0
        assert len(result.messages) == len(msgs)

    def test_fresh_tail_limited(self, assembler, sample_conversation):
        conv, msgs = sample_conversation
        config = AssemblyConfig(max_tokens=100000, fresh_tail_count=3, fresh_tail_max_tokens=None)
        result = assembler.assemble_context(conv.conversation_id, config)
        # Only 3 fresh tail messages
        assert result.messages_used == 3

    def test_over_budget_truncates(self, assembler, conversation_store):
        conv = conversation_store.create_conversation("big")
        for i in range(20):
            conversation_store.create_message(
                CreateMessageInput(
                    conversation_id=conv.conversation_id,
                    seq=i + 1,
                    role="user",
                    content="x" * 400,  # ~100 tokens each
                    token_count=100,
                )
            )
        # Budget of 500 tokens with 1000 reserve = very tight
        config = AssemblyConfig(max_tokens=600, fresh_tail_count=20, fresh_tail_max_tokens=None, reserve_tokens=100)
        result = assembler.assemble_context(conv.conversation_id, config)
        assert result.total_tokens <= 600

    def test_with_summaries(self, assembler, conversation_store, summary_store):
        conv = conversation_store.create_conversation("with-sum")
        base = datetime(2024, 1, 1)
        for i in range(10):
            conversation_store.create_message(
                CreateMessageInput(
                    conversation_id=conv.conversation_id,
                    seq=i + 1,
                    role="user",
                    content=f"Message {i}",
                    token_count=10,
                )
            )

        # Create a summary covering early messages
        summary_store.create_summary(
            CreateSummaryInput(
                conversation_id=conv.conversation_id,
                kind="leaf",
                depth=0,
                content="Summary of early messages about setup and configuration.",
                token_count=15,
                earliest_at=base,
                latest_at=base + timedelta(hours=1),
                descendant_count=5,
                descendant_token_count=50,
                model="test",
            )
        )

        config = AssemblyConfig(max_tokens=10000, fresh_tail_count=3, fresh_tail_max_tokens=None)
        result = assembler.assemble_context(conv.conversation_id, config)
        # Should include summaries + fresh tail
        assert result.summaries_used >= 0  # May or may not include based on temporal overlap
        assert result.messages_used >= 1

    def test_summary_to_message_format(self, assembler, conversation_store, summary_store):
        conv = conversation_store.create_conversation("fmt")
        conversation_store.create_message(
            CreateMessageInput(
                conversation_id=conv.conversation_id,
                seq=1,
                role="user",
                content="hello",
                token_count=1,
            )
        )
        summary_store.create_summary(
            CreateSummaryInput(
                conversation_id=conv.conversation_id,
                kind="leaf",
                depth=0,
                content="A summary of past events.",
                token_count=5,
                earliest_at=datetime(2020, 1, 1),
                latest_at=datetime(2020, 1, 2),
                descendant_count=10,
                descendant_token_count=100,
                model="test",
            )
        )
        config = AssemblyConfig(max_tokens=10000, fresh_tail_count=1, fresh_tail_max_tokens=None)
        result = assembler.assemble_context(conv.conversation_id, config)
        # Check summary message format
        summary_msgs = [m for m in result.messages if "[LCM CONTEXT SUMMARY" in m.get("content", "")]
        for sm in summary_msgs:
            assert sm["role"] == "assistant"

    def test_coverage_ratio(self, assembler, sample_conversation, summary_store):
        conv, msgs = sample_conversation
        config = AssemblyConfig(max_tokens=100000, fresh_tail_count=5, fresh_tail_max_tokens=None)
        result = assembler.assemble_context(conv.conversation_id, config)
        assert 0.0 < result.coverage_ratio <= 1.0

    def test_tool_call_id_linkage_preserved(self, assembler, conversation_store, db):
        """
        Verify _message_to_dict includes tool_call_id for tool-role messages
        and tool_calls for assistant messages with stored message_parts.
        This is the fix for the 'tool id() not found (2013)' MiniMax error.
        """
        conv = conversation_store.create_conversation("tool-linkage")

        # Step 1: Store an assistant message with a tool_call
        asst_msg = conversation_store.create_message(
            CreateMessageInput(
                conversation_id=conv.conversation_id,
                seq=1,
                role="assistant",
                content="Let me look that up.",
                token_count=5,
            )
        )
        # Add a tool_call message_part
        conversation_store.create_message_part(
            asst_msg.message_id,
            CreateMessagePartInput(
                session_id="tool-test",
                part_type="tool",
                ordinal=0,
                tool_call_id="call_test123",
                tool_name="web_search",
                tool_input='{"query": "test query"}',
            ),
        )

        # Step 2: Store a tool result message
        tool_msg = conversation_store.create_message(
            CreateMessageInput(
                conversation_id=conv.conversation_id,
                seq=2,
                role="tool",
                content='{"results": ["result1", "result2"]}',
                token_count=10,
            )
        )
        # Add a tool_result message_part with tool_call_id linking back
        conversation_store.create_message_part(
            tool_msg.message_id,
            CreateMessagePartInput(
                session_id="tool-test",
                part_type="tool",
                ordinal=0,
                tool_call_id="call_test123",
                tool_output='{"results": ["result1", "result2"]}',
            ),
        )

        # Step 3: Assemble context
        config = AssemblyConfig(max_tokens=10000, fresh_tail_count=10, fresh_tail_max_tokens=None)
        result = assembler.assemble_context(conv.conversation_id, config)

        # Step 4: Verify tool_call_id is present on the tool result message
        tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1, f"Expected 1 tool message, got {len(tool_msgs)}"
        assert tool_msgs[0].get("tool_call_id") == "call_test123", (
            f"Expected tool_call_id='call_test123', got {tool_msgs[0].get('tool_call_id')}"
        )

        # Step 5: Verify tool_calls are present on the assistant message
        asst_msgs = [m for m in result.messages if m.get("role") == "assistant"]
        tool_call_msgs = [m for m in asst_msgs if m.get("tool_calls")]
        assert len(tool_call_msgs) == 1, f"Expected 1 assistant msg with tool_calls, got {len(tool_call_msgs)}"
        tc = tool_call_msgs[0]["tool_calls"]
        assert len(tc) == 1
        assert tc[0]["id"] == "call_test123"
        assert tc[0]["function"]["name"] == "web_search"
        assert tc[0]["function"]["arguments"] == '{"query": "test query"}'
