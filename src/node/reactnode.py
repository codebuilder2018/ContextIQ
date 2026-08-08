"""LangGraph nodes for RAG workflow + ReAct Agent."""

from typing import List, Optional

from src.state.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun


class RAGNodes:
    """Contains node functions for the RAG workflow."""

    def __init__(self, retriever, llm):
        """
        Initialize RAG nodes.

        Args:
            retriever: Document retriever instance.
            llm: Language model instance.
        """
        self.retriever = retriever
        self.llm = llm
        self._agent = None

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents."""

        docs = self.retriever.invoke(state.question)

        return RAGState(
            question=state.question,
            retrieved_docs=docs,
        )

    def _build_tools(self):
        """Build retriever and Wikipedia tools."""

        # =========================================================
        # RETRIEVER TOOL
        # =========================================================

        def retriever_tool_fn(query: str) -> str:
            """
            Search the indexed document corpus.

            Args:
                query: Search query.

            Returns:
                Relevant document passages.
            """

            try:
                docs: List[Document] = self.retriever.invoke(query)

            except Exception as exc:
                return (
                    "Retriever failed while searching the indexed "
                    f"documents: {str(exc)}"
                )

            if not docs:
                return "No documents found in the indexed corpus."

            merged = []

            for i, doc in enumerate(docs[:8], start=1):
                metadata = getattr(doc, "metadata", {}) or {}

                title = (
                    metadata.get("title")
                    or metadata.get("source")
                    or f"document_{i}"
                )

                content = getattr(doc, "page_content", "")

                merged.append(
                    f"[{i}] {title}\n{content}"
                )

            return "\n\n".join(merged)

        retriever_tool = Tool(
            name="retriever",
            description=(
                "Search the indexed user document corpus and "
                "return relevant passages. Use this tool first "
                "when the question relates to the user's documents."
            ),
            func=retriever_tool_fn,
        )

        # =========================================================
        # WIKIPEDIA TOOL
        # =========================================================

        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(
                top_k_results=3,
                lang="en",
            )
        )

        def wikipedia_safe(query: str) -> str:
            """
            Safely query Wikipedia.

            Wikipedia/API errors should not crash the entire
            Agentic RAG workflow.
            """

            try:
                result = wiki.invoke(query)

                if not result:
                    return (
                        "Wikipedia returned no information for "
                        f"the query: {query}"
                    )

                return str(result)

            except Exception as exc:
                return (
                    "Wikipedia is currently unavailable. "
                    "Do not retry Wikipedia repeatedly. "
                    f"Error: {type(exc).__name__}: {str(exc)}"
                )

        wikipedia_tool = Tool(
            name="wikipedia_search",
            description=(
                "Search Wikipedia for general knowledge when "
                "the indexed document corpus does not contain "
                "the required information."
            ),
            func=wikipedia_safe,
        )

        return [
            retriever_tool,
            wikipedia_tool,
        ]

    def _build_agent(self):
        """Build the LangChain Agentic RAG agent."""

        tools = self._build_tools()

        system_prompt = (
            "You are a helpful Agentic RAG assistant. "

            "For questions related to the user's indexed "
            "documents, use the retriever tool first. "

            "Use the Wikipedia search tool only when general "
            "knowledge is needed or the indexed documents do "
            "not contain sufficient information. "

            "If a tool reports that it is unavailable, do not "
            "repeatedly call that tool. "

            "Do not invent facts. "

            "Base your answer on the information returned by "
            "the tools and the conversation. "

            "If sufficient information is unavailable, clearly "
            "say so. "

            "Return only the final useful answer to the user."
        )

        self._agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=system_prompt,
        )

    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate an answer using the Agentic RAG agent.

        Args:
            state: Current RAG state.

        Returns:
            Updated RAG state containing the final answer.
        """

        if self._agent is None:
            self._build_agent()

        try:
            result = self._agent.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content=state.question
                        )
                    ]
                }
            )

            messages = result.get("messages", [])

            answer: Optional[str] = None

            if messages:
                answer_message = messages[-1]
                answer = getattr(
                    answer_message,
                    "content",
                    None,
                )

            return RAGState(
                question=state.question,
                retrieved_docs=state.retrieved_docs,
                answer=answer or "Could not generate an answer.",
            )

        except Exception as exc:
            return RAGState(
                question=state.question,
                retrieved_docs=state.retrieved_docs,
                answer=(
                    "I could not generate the answer because "
                    f"the agent encountered an error: "
                    f"{type(exc).__name__}: {str(exc)}"
                ),
            )