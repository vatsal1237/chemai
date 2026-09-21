"""
RAG Chain — the main orchestrator.
Manages conversation history, retrieval, LLM calls, and the RETRIEVE_WITH_CONTEXT
re-retrieval loop from the decision rules.
"""

import re
from dataclasses import dataclass, field

from config.settings import (
    RAG_SYSTEM_PROMPT,
    RAG_USER_TEMPLATE,
    MAX_HISTORY_TURNS,
)
from core.vectorstore import VectorStore
from core.retriever import retrieve
from core.llm import query_llm
import re
from config.settings import LLM_MODEL, LLM_TEMPERATURE


@dataclass
class ChatMessage:
    """A single message in the conversation."""
    role: str  # "user" or "assistant"
    content: str


class RAGChain:
    """
    Full RAG pipeline orchestrator.

    Implements the multi-step decision-rule prompt:
    1. First retrieval with user's raw query
    2. If LLM responds with RETRIEVE_WITH_CONTEXT, re-retrieves with rewritten query
    3. Returns final answer
    """

    def __init__(self, store: VectorStore):
        self.store = store
        self.history: list[ChatMessage] = []

    def ask(self, user_query: str) -> str:
        """
        Process a user query through the full RAG pipeline.

        Args:
            user_query: The user's question.

        Returns:
            The assistant's answer string.
        """
        # Step 1: Retrieve relevant chunks
        context_str, hits = retrieve(self.store, user_query)

        # Step 2: Build the prompt
        chat_history = self._format_history()
        prompt = RAG_USER_TEMPLATE.format(
            chat_history=chat_history if chat_history else "No previous conversation.",
            retrieved_chunks=context_str,
            user_query=user_query,
        )

        # Step 3: Query the LLM
        answer = query_llm(prompt, system_prompt=RAG_SYSTEM_PROMPT)

        # Step 4: Handle RETRIEVE_WITH_CONTEXT re-retrieval
        answer = self._handle_re_retrieval(answer, user_query, chat_history)

        # Step 5: Update history
        self.history.append(ChatMessage(role="user", content=user_query))
        self.history.append(ChatMessage(role="assistant", content=answer))

        # Trim history to max turns
        max_messages = MAX_HISTORY_TURNS * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

        return answer

    def ask_stream(self, user_query: str, model: str | None = None) -> Generator[str, None, None]:
        """
        Ask a question and stream the answer token-by-token.

        Note: Re-retrieval is handled non-streaming since we need
        the full response to detect RETRIEVE_WITH_CONTEXT.

        Args:
            user_query: The user's question.
            model: Optional Gemini model override.

        Yields:
            Response tokens as strings.
        """
        # Step 1: Retrieve
        context_str, hits = retrieve(self.store, user_query)

        # Step 2: Build prompt
        chat_history = self._format_history()
        prompt = RAG_USER_TEMPLATE.format(
            chat_history=chat_history if chat_history else "No previous conversation.",
            retrieved_chunks=context_str,
            user_query=user_query,
        )

        # Step 3: First LLM call — non-streaming to check for re-retrieval
        first_answer = query_llm(prompt, system_prompt=RAG_SYSTEM_PROMPT, model=model)

        # Step 4: Check for re-retrieval
        re_retrieval_match = re.search(
            r'RETRIEVE_WITH_CONTEXT:\s*"([^"]+)"', first_answer
        )

        if re_retrieval_match:
            rewritten_query = re_retrieval_match.group(1)
            print(f"[+] Re-retrieving with: '{rewritten_query}'", flush=True)

            new_context, _ = retrieve(self.store, rewritten_query)
            new_prompt = RAG_USER_TEMPLATE.format(
                chat_history=chat_history if chat_history else "No previous conversation.",
                retrieved_chunks=new_context,
                user_query=user_query,
            )

            # Stream the final answer using Gemini
            full_response = []
            
            for token in query_llm(new_prompt, system_prompt=RAG_SYSTEM_PROMPT, model=model, stream=True):
                full_response.append(token)
                yield token

            final_answer = "".join(full_response)
        else:
            # No re-retrieval needed — yield first answer token by token
            final_answer = first_answer
            # Simulate streaming by yielding word by word for smooth UX
            words = first_answer.split(" ")
            for i, word in enumerate(words):
                if i < len(words) - 1:
                    yield word + " "
                else:
                    yield word

        # Update history
        self.history.append(ChatMessage(role="user", content=user_query))
        self.history.append(ChatMessage(role="assistant", content=final_answer))

        max_messages = MAX_HISTORY_TURNS * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history = []

    def _format_history(self) -> str:
        """Format conversation history for the prompt template."""
        if not self.history:
            return ""

        lines = []
        for msg in self.history:
            role = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{role}: {msg.content}")

        return "\n".join(lines)

    def _handle_re_retrieval(
        self, answer: str, original_query: str, chat_history: str
    ) -> str:
        """
        Check if LLM requested a re-retrieval via RETRIEVE_WITH_CONTEXT.
        If so, perform a second retrieval with the rewritten query and re-query the LLM.
        """
        match = re.search(r'RETRIEVE_WITH_CONTEXT:\s*"([^"]+)"', answer)
        if not match:
            return answer

        rewritten_query = match.group(1)
        print(f"[+] Re-retrieval triggered with: '{rewritten_query}'", flush=True)

        # Second retrieval
        new_context, _ = retrieve(self.store, rewritten_query)

        # Re-query LLM with new context
        new_prompt = RAG_USER_TEMPLATE.format(
            chat_history=chat_history if chat_history else "No previous conversation.",
            retrieved_chunks=new_context,
            user_query=original_query,
        )

        final_answer = query_llm(new_prompt, system_prompt=RAG_SYSTEM_PROMPT)

        # Prevent infinite re-retrieval loops
        if "RETRIEVE_WITH_CONTEXT:" in final_answer:
            return (
                "I attempted to find more relevant information in the document, "
                "but couldn't find sufficient context to answer your question. "
                "Could you rephrase or ask about a specific topic from the paper?"
            )

        return final_answer
