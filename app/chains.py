import logging
import os

from app.llm import get_llm
from app.prompts import chat_prompt
from app.memory import get_session_history

logger = logging.getLogger(__name__)

from langchain_core.runnables import  RunnableWithMessageHistory



def build_chain():

    try:
        llm = get_llm()

        chain = chat_prompt | llm

        chain_with_memory = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )

        return chain_with_memory


    except Exception:
        logger.exception("Chain build failed")
        raise RuntimeError("AI chain unavailable")