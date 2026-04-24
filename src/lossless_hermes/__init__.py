"""LCM Context Engine Plugin for Hermes Agent.

Provides lossless context management with DAG-based summarization,
cache-aware compaction, and full-text search capabilities.
"""

__version__ = "0.1.5"

import logging
from typing import Any

# Import Hermes Agent's context engine interface (optional dependency)
try:
    from agent.context_engine import ContextEngine
except ImportError:
    # Running standalone (not as a Hermes Agent plugin) — provide a base class stub
    class ContextEngine:
        """Stub ContextEngine for standalone usage."""

        def update_from_response(self, usage):
            pass

        def should_compress(self, prompt_tokens=None):
            return False

        def compress(self, messages, current_tokens=None, **kwargs):
            return messages

        def on_session_start(self, session_id, **kwargs):
            pass

        def on_session_end(self, session_id, messages):
            pass

        def on_session_reset(self):
            pass

        def get_tool_schemas(self):
            return []

        def handle_tool_call(self, name, args, **kwargs):
            return "{}"

        def get_status(self):
            return {
                "name": getattr(self, "name", "unknown"),
                "threshold_tokens": getattr(self, "threshold_tokens", 0),
                "context_length": getattr(self, "context_length", 0),
                "compression_count": getattr(self, "compression_count", 0),
            }

        def update_model(self, model, context_length, **kwargs):
            pass


# Import LCM components
from .assembler import AssemblyConfig, ContextAssembler
from .compaction import CompactionConfig, CompactionEngine, CompactionResult
from .db.config import LcmConfig, resolve_lcm_config
from .db.connection import close_database, get_database, initialize_database
from .db.migration import run_lcm_migrations
from .retrieval import RetrievalEngine
from .store.conversation import ConversationStore, CreateMessageInput, MessageRecord
from .store.summary import SummaryStore
from .summarizer import create_lcm_summarizer
from .tokens import estimate_messages_tokens, estimate_tokens
from .tools import LcmTools, get_tool_schemas

__all__ = [
    "LcmContextEngine",
    "LcmConfig",
    "CompactionResult",
    "MessageRecord",
    "LcmTools",
    "get_tool_schemas",
    "close_database",
    "get_database",
    "estimate_messages_tokens",
    "estimate_tokens",
    "register",
]

logger = logging.getLogger(__name__)


class LcmContextEngine(ContextEngine):
    """LCM (Lossless Context Management) Context Engine for Hermes Agent."""

    @property
    def name(self) -> str:
        return "lossless-hermes"

    def __init__(
        self,
        model: str = "",
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
        config_context_length: int | None = None,
        plugin_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Initialize the LCM context engine."""

        # Store runtime parameters
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.provider = provider
        self.api_mode = api_mode

        # Load configuration — try plugin.yaml co-located with this module
        if plugin_config is None:
            import os

            import yaml

            _yaml_path = os.path.join(os.path.dirname(__file__), "plugin.yaml")
            if os.path.exists(_yaml_path):
                try:
                    with open(_yaml_path) as _f:
                        _meta = yaml.safe_load(_f) or {}
                    plugin_config = _meta.get("config", {})
                except Exception:
                    plugin_config = {}
        self.config = resolve_lcm_config(plugin_config=plugin_config)

        # Initialize context length and thresholds
        self.context_length = config_context_length or 128000  # Default to reasonable size
        self.threshold_tokens = int(self.context_length * self.config.context_threshold)

        # Initialize state tracking
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0

        # Session state
        self.current_conversation_id: int | None = None
        self.current_session_id: str | None = None

        # Initialize stores and engines (will be set up on first use)
        self.conversation_store: ConversationStore | None = None
        self.summary_store: SummaryStore | None = None
        self.compaction_engine: CompactionEngine | None = None
        self.assembler: ContextAssembler | None = None
        self.retrieval_engine: RetrievalEngine | None = None
        self.tools: LcmTools | None = None

        # LLM call function (will be injected)
        self.call_llm_fn = None

        logger.debug(f"LCM context engine constructed: model={model} provider={provider}")

    def _ensure_initialized(self):
        """Ensure database and stores are initialized."""
        if not self.config.enabled:
            raise RuntimeError("LCM is disabled in configuration")

        if self.conversation_store is None:
            # Initialize database
            db = initialize_database(self.config)

            # Run migrations
            run_lcm_migrations(db)

            # Initialize stores
            self.conversation_store = ConversationStore(db)
            self.summary_store = SummaryStore(db)

            # Initialize engines
            summarizer = create_lcm_summarizer(
                provider=self.config.summary_provider or self.provider,
                model=self.config.summary_model or self.model,
                timeout_ms=self.config.summary_timeout_ms,
                custom_instructions=self.config.custom_instructions,
                circuit_breaker_threshold=self.config.circuit_breaker_threshold,
                circuit_breaker_cooldown_ms=self.config.circuit_breaker_cooldown_ms,
                call_llm_fn=self.call_llm_fn,
            )

            compaction_config = CompactionConfig(
                leaf_chunk_tokens=self.config.leaf_chunk_tokens,
                leaf_target_tokens=self.config.leaf_target_tokens,
                condensed_target_tokens=self.config.condensed_target_tokens,
                leaf_min_fanout=self.config.leaf_min_fanout,
                condensed_min_fanout=self.config.condensed_min_fanout,
                condensed_min_fanout_hard=self.config.condensed_min_fanout_hard,
                incremental_max_depth=self.config.incremental_max_depth,
                fresh_tail_count=self.config.fresh_tail_count,
                fresh_tail_max_tokens=self.config.fresh_tail_max_tokens,
                cache_aware_compaction=self.config.cache_aware_compaction,
            )

            self.compaction_engine = CompactionEngine(
                compaction_config, summarizer, self.conversation_store, self.summary_store
            )

            self.assembler = ContextAssembler(self.conversation_store, self.summary_store)

            self.retrieval_engine = RetrievalEngine(self.conversation_store, self.summary_store)

            logger.info("LCM components initialized successfully")

    def _ensure_conversation(self, session_id: str) -> int:
        """Ensure conversation exists for session and return conversation_id."""
        self._ensure_initialized()

        if self.current_session_id == session_id and self.current_conversation_id:
            return self.current_conversation_id

        # Get or create conversation
        conversation = self.conversation_store.get_conversation_by_session(session_id)

        if not conversation:
            conversation = self.conversation_store.create_conversation(session_id)
            logger.info(f"Created new conversation {conversation.conversation_id} for session {session_id}")

        self.current_session_id = session_id
        self.current_conversation_id = conversation.conversation_id

        # Initialize tools for this conversation
        self.tools = LcmTools(conversation.conversation_id)

        return conversation.conversation_id

    def update_from_response(self, usage: dict[str, Any]) -> None:
        """Update tracked token usage from an API response."""
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        """Check if context exceeds the compression threshold."""
        if not self.config.enabled:
            return False

        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        return tokens >= self.threshold_tokens

    def compress(self, messages: list[dict[str, Any]], current_tokens: int = None, **kwargs) -> list[dict[str, Any]]:
        """Compact the message list using LCM's DAG-based approach."""
        if not self.config.enabled:
            return messages

        try:
            self._ensure_initialized()

            # Use current session if available, fall back to kwargs
            session_id = self.current_session_id or kwargs.get("session_id", "default_session")
            conversation_id = self.current_conversation_id or self._ensure_conversation(session_id)

            # Ingest new messages
            self._ingest_messages(conversation_id, messages)

            # Estimate current tokens if not provided
            if current_tokens is None:
                current_tokens = estimate_messages_tokens(messages)

            # Run compaction
            cache_state = kwargs.get("cache_state", "unknown")
            should_compact, reason = self.compaction_engine.should_compact(
                conversation_id, current_tokens, self.threshold_tokens, cache_state
            )

            if should_compact:
                logger.info(f"Running LCM compaction: {reason}")

                result = self.compaction_engine.compact(conversation_id, current_tokens, cache_state)

                self.compression_count += 1

                logger.info(
                    f"LCM compaction complete: {result.summaries_created} summaries, {result.tokens_saved} tokens saved"
                )

            # Assemble optimal context
            assembly_config = AssemblyConfig(
                max_tokens=self.context_length,
                fresh_tail_count=self.config.fresh_tail_count,
                fresh_tail_max_tokens=self.config.fresh_tail_max_tokens,
            )

            assembly_result = self.assembler.assemble_context(conversation_id, assembly_config)

            logger.info(
                f"LCM context assembled: {assembly_result.messages_used} messages, "
                f"{assembly_result.summaries_used} summaries, "
                f"{assembly_result.total_tokens} tokens"
            )

            return assembly_result.messages

        except Exception as e:
            logger.error(f"LCM compression failed: {e}")
            # Fallback to original messages
            return messages

    def _ingest_messages(self, conversation_id: int, messages: list[dict[str, Any]]):
        """Ingest new messages into LCM storage.

        Safe for progressive/repeated calls - uses identity hashing to
        deduplicate messages that have already been stored.
        """
        if not messages:
            return

        import hashlib

        # Get existing identity hashes to skip duplicates
        try:
            existing = self.conversation_store.db.execute(
                "SELECT identity_hash FROM messages WHERE conversation_id = ? AND identity_hash IS NOT NULL",
                (conversation_id,),
            ).fetchall()
            existing_hashes = {row[0] for row in existing}
        except Exception:
            existing_hashes = set()

        # Get current message count to determine sequence numbers
        latest_seq = self.conversation_store.get_latest_message_seq(conversation_id)
        added = 0

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            content_str = str(content)

            # Compute identity hash for dedup
            identity = hashlib.sha256(f"{role}:{content_str}".encode()).hexdigest()[:16]
            if identity in existing_hashes:
                continue

            token_count = estimate_tokens(content_str)

            msg_input = CreateMessageInput(
                conversation_id=conversation_id,
                seq=latest_seq + added + 1,
                role=role,
                content=content_str,
                token_count=token_count,
                identity_hash=identity,
            )

            try:
                self.conversation_store.create_message(msg_input)
                existing_hashes.add(identity)
                added += 1
            except Exception as e:
                # Message might already exist, skip it
                logger.debug(f"Skipping message ingestion: {e}")
                continue

    def on_session_start(self, session_id: str, **kwargs) -> None:
        """Called when a new conversation session begins."""
        try:
            self._ensure_conversation(session_id)
            logger.info(f"LCM session started: {session_id}")
        except Exception as e:
            logger.error(f"LCM session start failed: {e}")

    def on_session_end(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Called at session boundaries."""
        try:
            if self.current_session_id == session_id:
                # Final ingestion of any remaining messages
                if self.current_conversation_id and messages:
                    self._ingest_messages(self.current_conversation_id, messages)

                # Reset session state
                self.current_session_id = None
                self.current_conversation_id = None
                self.tools = None

            logger.info(f"LCM session ended: {session_id}")
        except Exception as e:
            logger.error(f"LCM session end failed: {e}")

    def on_session_reset(self) -> None:
        """Called on /new or /reset."""
        try:
            super().on_session_reset()
        except (AttributeError, TypeError):
            pass
        self.current_session_id = None
        self.current_conversation_id = None
        self.tools = None

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return tool schemas for LCM tools."""
        if not self.config.enabled:
            return []

        return get_tool_schemas()

    def handle_tool_call(self, name: str, args: dict[str, Any], **kwargs) -> str:
        """Handle a tool call from the agent."""
        if not self.config.enabled:
            return '{"error": "LCM is disabled"}'

        try:
            # Use current session if available, fall back to kwargs
            session_id = self.current_session_id or kwargs.get("session_id", "default_session")
            conversation_id = self.current_conversation_id or self._ensure_conversation(session_id)

            # Progressive ingestion: ingest messages before handling tool calls
            # so that grep/describe/expand have current data
            messages = kwargs.get("messages")
            if messages and conversation_id:
                try:
                    self._ensure_initialized()
                    self._ingest_messages(conversation_id, messages)
                except Exception as e:
                    logger.debug(f"Progressive ingestion before tool call failed: {e}")

            # Initialize tools if not already done
            if not self.tools:
                self.tools = LcmTools(conversation_id)

            return self.tools.handle_tool_call(name, args)

        except Exception as e:
            logger.error(f"LCM tool call failed: {e}")
            return f'{{"error": "Tool call failed: {str(e)}"}}'

    def get_status(self) -> dict[str, Any]:
        """Return status dict for display/logging."""
        try:
            status = super().get_status()
        except (AttributeError, TypeError):
            status = {
                "name": self.name,
                "threshold_tokens": self.threshold_tokens,
                "context_length": self.context_length,
                "compression_count": self.compression_count,
            }

        if self.config.enabled and self.current_conversation_id:
            try:
                # Add LCM-specific status
                stats = self.summary_store.get_summary_depth_stats(self.current_conversation_id)
                messages = self.conversation_store.get_messages_by_conversation(self.current_conversation_id)

                status.update(
                    {
                        "lcm_enabled": True,
                        "conversation_id": self.current_conversation_id,
                        "message_count": len(messages),
                        "summary_stats": stats,
                        "lcm_config": {
                            "leaf_chunk_tokens": self.config.leaf_chunk_tokens,
                            "fresh_tail_count": self.config.fresh_tail_count,
                            "context_threshold": self.config.context_threshold,
                        },
                    }
                )
            except Exception as e:
                status["lcm_error"] = str(e)
        else:
            status["lcm_enabled"] = self.config.enabled

        return status

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
    ) -> None:
        """Called when the user switches models."""
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.provider = provider
        self.api_mode = api_mode
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self.config.context_threshold)

        logger.info(f"LCM model updated: {model} (context: {context_length}, threshold: {self.threshold_tokens})")


def register(ctx):
    """Register the LCM context engine plugin."""
    engine = LcmContextEngine(model="", provider="")
    ctx.register_context_engine(engine)
    logger.info("LCM context engine plugin registered")
