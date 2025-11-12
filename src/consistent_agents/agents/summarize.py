import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from litellm.utils import get_max_tokens, token_counter

from consistent_agents.agents.default import AgentConfig, CoTAgentConfig, DefaultAgent
from consistent_agents.environments.base import BaseEnvironment
from consistent_agents.models.base import BaseModel

@dataclass
class SummarizeAgentConfig(AgentConfig):
    """
    Extends AgentConfig with summarization-specific parameters.
    """
    # Trigger summarization when dropped below this value
    proactive_summarization_threshold: int = 8000
    target_free_tokens_after_unwind: int = 4000

    # Summarization prompt templates
    summary_request_template: str = (
        "You are about to hand off your work to another AI agent. "
        "Please provide a comprehensive summary of what you have accomplished so far on this task:\n"
        "Original Task: {{task}}\n\n"
        "Based on the conversation history, please provide a detailed summary covering:\n"
        "1. **Major Actions Completed** - List each significant command you executed and what you learned from it.\n"
        "2. **Important Information Learned** - A summary of crucial findings, file locations, configurations, error messages, or system state discovered.\n"
        "3. **Challenging Problems Addressed** - Any significant issues you encountered and how you resolved them.\n"
        "4. **Current Status** - Exactly where you are in the task completion process.\n\n"
        "Be comprehensive and detailed. The next agent needs to understand everything that has happened so far in order to continue."
    )
    
    questions_request_template: str = (
        "You are picking up work from a previous AI agent on this task:\n\n"
        "**Original Task:**\n{{task}}\n\n"
        "**Summary from Previous Agent:**\n{{summary}}\n\n"
        "Please begin by asking five questions, or more if necessary, about the current state of the solution that are not answered in the summary from the prior agent. "
        "After you ask these questions you will be on your own, so ask everything you need to know."
    )
    
    answers_request_template: str = (
        "The next agent has a few questions for you, please answer each of them one by one in detail:\n\n"
        "{{questions}}"
    )
    
    continuation_template: str = (
        "Here are the answers the other agent provided:\n\n{{answers}}\n\n"
        "Continue working on this task from where the previous agent left off. "
        "You can no longer ask questions. Please follow the spec to interact with "
        "the environment."
    )
    
    
class SummarizeAgent(DefaultAgent):
    """
    Agent that automatically summarizes conversation history when context limit is approached.
    
    This agent monitors token usage and performs proactive summarization using a Q&A
    handoff pattern that preserves critical task information. It handles both proactive
    summarization (before hitting limits) and reactive summarization (when errors occur).
    
    The summarization process:
    1. "Old agent" summarizes its work
    2. "New agent" asks clarifying questions about gaps
    3. "Old agent" answers the questions
    4. Message history is reconstructed with the Q&A continuation
    """
    
    def __init__(
        self,
        model: BaseModel,
        *,
        config_class_name: str = "default",
        proactive_summarization_threshold: int = 8000,
        target_free_tokens_after_unwind: int = 4000,
        **kwargs
    ):
        """
        Initialize the SummarizeAgent.
        
        Args:
            model: BaseModel instance (e.g., LitellmModel)
            config_class_name: "default" or "cot" 
            proactive_summarization_threshold: Free tokens before triggering summarization
            target_free_tokens_after_unwind: Target free tokens after unwinding messages
            **kwargs: Additional config params (step_limit, cost_limit, etc.)
        """
        if config_class_name == "default":
            config_class = SummarizeAgentConfig
        elif config_class_name == "cot":
            @dataclass
            class CoTSummarizeAgentConfig(SummarizeAgentConfig, CoTAgentConfig):
                """Combined CoT and Summarization configuration."""
                pass
            config_class = CoTSummarizeAgentConfig
        else:
            raise ValueError(
                f"Unknown config_class_name: {config_class_name}. "
                "Use 'default' or 'cot'."
            )
        
        summary_params = {
            'proactive_summarization_threshold': proactive_summarization_threshold,
            'target_free_tokens_after_unwind': target_free_tokens_after_unwind,
        }
        all_params = {**kwargs, **summary_params}
        self.config = config_class(**all_params)
        self.messages: List[Dict[str, Any]] = []
        self.model = model
        self.extra_template_vars: Dict[str, Any] = {}
        self.steps: int = 0
        # Summarization-specific stats
        self.summarization_count: int = 0
        self.total_tokens_before_summaries: int = 0
        self.total_tokens_after_summaries: int = 0
        self.last_summarization_step: int = -1
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def run(self, task: str, env: BaseEnvironment, **kwargs) -> Tuple[str, str]:
        """Overrides parent to store env reference for summarization methods."""
        self.env = env
        return super().run(task, env, **kwargs)
    
    def _count_total_tokens(self) -> int:
        """
        Count total tokens in current message history using litellm's token_counter.
        
        Returns:
            int: Total tokens across all messages
        """
        try:
            return token_counter(
                model=self.model.config.model_name,
                messages=self.messages
            )
        except Exception as e:
            self.logger.warning(f"Token counting failed: {e}. Using fallback estimate.")
            total_chars = sum(len(str(msg.get("content", ""))) for msg in self.messages)
            return total_chars // 4
    
    def _get_model_context_limit(self) -> int:
        """
        Get the context window size for the current model.
        
        Returns:
            int: Maximum tokens (defaults to 1M if unknown)
        """
        fallback_limit = 1_000_000
        try:
            max_tokens = get_max_tokens(self.model.config.model_name)
            if max_tokens is None:
                self.logger.warning(
                    f"Could not determine context limit for {self.model.config.model_name}. "
                    f"Using fallback: {fallback_limit}"
                )
                return fallback_limit
            return max_tokens
        except Exception as e:
            self.logger.warning(
                f"Error getting context limit: {e}. Using fallback: {fallback_limit}"
            )
            return fallback_limit
    
    def _get_free_tokens(self) -> int:
        """
        Calculate remaining tokens before hitting context limit.
        
        Returns:
            int: Free tokens available
        """
        context_limit = self._get_model_context_limit()
        current_tokens = self._count_total_tokens()
        return context_limit - current_tokens
    
    
    def _should_summarize(self) -> bool:
        """
        Determine if summarization should be triggered.
        
        Summarization is triggered when:
        - Free tokens drop below proactive_summarization_threshold
        - We have more than just the initial messages
        - We haven't summarized in the last few steps (avoid loops)
        
        Returns:
            bool: True if summarization should occur
        """
        # Need at least a few messages to summarize
        if len(self.messages) <= 2:
            return False
        
        # Avoid summarizing too frequently (wait at least 3 steps)
        if self.steps - self.last_summarization_step < 3:
            return False
        
        free_tokens = self._get_free_tokens()
        should_trigger = free_tokens < self.config.proactive_summarization_threshold
        
        if should_trigger:
            self.logger.info(
                f"Summarization triggered at step {self.steps}. "
                f"Free tokens: {free_tokens} / {self.config.proactive_summarization_threshold}"
            )
        
        return should_trigger
    
    
    def _unwind_messages_to_free_tokens(
        self,
        target_free_tokens: int | None = None
    ) -> None:
        """
        Remove recent message pairs (user+assistant) until enough tokens are freed.
        
        Always preserves the first message (system prompt + initial task).
        Removes messages in pairs to maintain conversation structure.
        
        Args:
            target_free_tokens: Desired free tokens (uses config default if None)
        """
        if target_free_tokens is None:
            target_free_tokens = self.config.target_free_tokens_after_unwind
        
        context_limit = self._get_model_context_limit()
        initial_count = len(self.messages)
        
        # Keep unwinding until we have enough free tokens
        while len(self.messages) > 1:
            current_tokens = self._count_total_tokens()
            free_tokens = context_limit - current_tokens
            
            if free_tokens >= target_free_tokens:
                break
            
            # Remove last 2 messages (user and assistant pair)
            if len(self.messages) >= 3:  # Keep at least system message
                self.messages = self.messages[:-2]
            else:
                break
        
        removed_count = initial_count - len(self.messages)
        final_free = self._get_free_tokens()
        
        if removed_count > 0:
            self.logger.info(
                f"Unwound {removed_count} messages. "
                f"Remaining: {len(self.messages)} messages, ~{final_free} free tokens"
            )
    
    def _get_current_env_state(self) -> str:
        """
        Get current state snapshot from environment.
        
        Attempts multiple strategies to capture environment state:
        1. Direct state getter if available
        2. Execute simple commands (pwd, ls)
        3. Fallback to placeholder
        
        Returns:
            str: Current environment state description
        """
        if self.env is None:
            return "[Environment not initialized]"
        
        try:
            # Try direct state getter
            if hasattr(self.env, 'get_current_state'):
                return self.env.get_current_state()
            
            # Try executing status commands
            if hasattr(self.env, 'execute'):
                try:
                    result = self.env.execute("pwd && ls -la 2>/dev/null || echo 'No files'")
                    output = result.get('output', '')
                    if output:
                        return output
                except Exception as e:
                    self.logger.debug(f"Could not execute status command: {e}")
            
            # Fallback
            return "[Environment state capture not available]"
            
        except Exception as e:
            self.logger.warning(f"Error capturing environment state: {e}")
            return f"[Error getting environment state: {e}]"

    
    def _create_summary_from_history(self) -> str:
        """
        Generate summary using Q&A handoff pattern from terminus_2.
        
        This creates a "handoff" between the "old agent" and "new agent":
        1. Old agent summarizes its work
        2. New agent asks clarifying questions
        3. Old agent answers the questions
        4. New agent receives answers to continue
        
        Returns:
            str: Continuation prompt containing Q&A for new "agent"
        """
        self.logger.info("Starting Q&A handoff summarization...")
        
        # Step 1: Ask current "agent" to summarize its work
        summary_prompt = self.render_template(
            self.config.summary_request_template
        )
        
        # Query model with full history to generate summary
        summary_response = self.model.query(
            self.messages + [{"role": "user", "content": summary_prompt}]
        )
        summary_text = summary_response["content"]
        self.logger.debug(f"Generated summary ({len(summary_text)} chars)")
        
        # Step 2: "New agent" asks questions about gaps in summary
        current_state = self._get_current_env_state()
        questions_prompt = self.render_template(
            self.config.questions_request_template,
            summary=summary_text,
            current_state=current_state
        )
        
        questions_response = self.model.query([
            {"role": "user", "content": questions_prompt}
        ])
        questions_text = questions_response["content"]
        self.logger.debug(f"Generated questions ({len(questions_text)} chars)")
        
        # Step 3: "Old agent" answers the questions
        answers_prompt = self.render_template(
            self.config.answers_request_template,
            questions=questions_text
        )
        
        # Query WITH history so old agent can answer accurately
        answers_response = self.model.query(
            self.messages + [{"role": "user", "content": answers_prompt}]
        )
        answers_text = answers_response["content"]
        self.logger.debug(f"Generated answers ({len(answers_text)} chars)")
        
        # Step 4: Create continuation prompt for "new agent" with answers
        continuation_prompt = self.render_template(
            self.config.continuation_template,
            answers=answers_text
        )
        
        self.logger.info("Q&A handoff complete")
        return continuation_prompt
    
    def _perform_summarization(self) -> None:
        """
        Execute the complete summarization process.
        
        Process:
        1. Track tokens before summarization
        2. Unwind messages to free space for summarization queries
        3. Generate summary via Q&A handoff
        4. Reconstruct message history with summary
        5. Track metrics for evaluation
        
        Raises:
            Exception: If summarization fails 
        """
        # Track metrics
        tokens_before = self._count_total_tokens()
        self.total_tokens_before_summaries += tokens_before
        
        self.logger.info(
            f"Starting summarization #{self.summarization_count + 1} at step {self.steps}. "
            f"Current tokens: {tokens_before}"
        )
        
        try:
            # Unwind to make room for summarization queries
            # (The Q&A process will add temporary tokens)
            self._unwind_messages_to_free_tokens()
            
            # Generate summary using Q&A handoff pattern
            summary_continuation = self._create_summary_from_history()
            
            # Reconstruct message history
            # Keep: [first message (system + task)] + [summary continuation]
            if len(self.messages) < 1:
                raise ValueError("Cannot summarize: no messages to preserve")
            
            first_message = self.messages[0]  # System message with task
            
            # Build new message list
            new_messages = [
                first_message,
                {
                    "role": "user",
                    "content": summary_continuation
                }
            ]
            
            # Replace message history
            self.messages = new_messages
            
            # Track metrics
            tokens_after = self._count_total_tokens()
            self.total_tokens_after_summaries += tokens_after
            self.summarization_count += 1
            self.last_summarization_step = self.steps
            
            tokens_saved = tokens_before - tokens_after
            compression_ratio = (tokens_after / tokens_before * 100) if tokens_before > 0 else 0
            
            self.logger.info(
                f"Summarization complete. "
                f"Tokens: {tokens_before:,} → {tokens_after:,} "
                f"(saved {tokens_saved:,}, {compression_ratio:.1f}% of original)"
            )
            
        except Exception as e:
            self.logger.error(f"Summarization failed at step {self.steps}: {e}", exc_info=True)
            raise
    
    def step(self) -> Dict[str, Any]:
        """
        Execute one step of the agent loop with proactive summarization.
        
        Checks if summarization should be triggered BEFORE querying the model.
        This prevents context length errors during normal operation.
        
        Returns:
            Dict[str, Any]: Observation from executing the action
        """
        # Check if we should summarize BEFORE querying model
        if self._should_summarize():
            try:
                self._perform_summarization()
            except Exception as e:
                self.logger.error(f"Proactive summarization failed: {e}")
                # If summarization fails, try unwinding as fallback
                try:
                    self._unwind_messages_to_free_tokens()
                except Exception as unwind_error:
                    self.logger.error(f"Fallback unwinding also failed: {unwind_error}")
                    # Continue anyway - let the query handle it
        
        # Continue with normal step from parent class
        return super().step()
    
    def query(self) -> Dict[str, Any]:
        """
        Query the model with reactive summarization on context errors.
        
        This catches context length exceeded errors that slip through proactive
        checks (e.g., if model response is very long). Falls back to forced
        summarization and retry.
        
        Returns:
            Dict[str, Any]: Model response
            
        Raises:
            Exception: Re-raises non-context-length errors
        """
        try:
            # Try normal query
            return super().query()
            
        except Exception as e:
            # Check if it's a context length error
            error_type = type(e).__name__
            error_msg = str(e).lower()
            
            is_context_error = (
                "context" in error_type.lower() or
                "context" in error_msg or
                "length" in error_msg and "exceed" in error_msg or
                "too long" in error_msg
            )
            
            if is_context_error:
                self.logger.warning(
                    f"Context length error caught during query: {e}. "
                    "Forcing emergency summarization."
                )
                
                try:
                    # Emergency: unwind and summarize immediately
                    self._unwind_messages_to_free_tokens()
                    self._perform_summarization()
                    
                    # Retry query after summarization
                    self.logger.info("Retrying query after emergency summarization")
                    return super().query()
                    
                except Exception as summary_error:
                    self.logger.error(
                        f"Emergency summarization failed: {summary_error}. "
                        "Cannot recover from context length error."
                    )
                    raise
            else:
                # Not a context error - re-raise
                raise
    
    def get_template_vars(self) -> Dict[str, Any]:
        """
        Get template variables for logging and debugging.
        
        Adds summarization-specific metrics to base template vars.
        
        Returns:
            Dict[str, Any]: Template variables including summarization stats
        """
        # Get base variables if parent has this method
        base_vars = {}
        if hasattr(super(), 'get_template_vars'):
            try:
                base_vars = super().get_template_vars()
            except Exception:
                pass
        
        # Calculate summarization metrics
        tokens_saved_total = (
            self.total_tokens_before_summaries - self.total_tokens_after_summaries
        )
        
        avg_tokens_before = (
            self.total_tokens_before_summaries / self.summarization_count
            if self.summarization_count > 0 else 0
        )
        
        avg_tokens_after = (
            self.total_tokens_after_summaries / self.summarization_count
            if self.summarization_count > 0 else 0
        )
        
        avg_compression_ratio = (
            (avg_tokens_after / avg_tokens_before * 100)
            if avg_tokens_before > 0 else 0
        )
        
        # Return merged variables
        return {
            **base_vars,
            "summarization_count": self.summarization_count,
            "tokens_saved_total": tokens_saved_total,
            "avg_tokens_before_summary": int(avg_tokens_before),
            "avg_tokens_after_summary": int(avg_tokens_after),
            "avg_compression_ratio_pct": round(avg_compression_ratio, 1),
            "last_summarization_step": self.last_summarization_step,
        }