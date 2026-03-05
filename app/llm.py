import os
import logging
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))


def get_llm():
    """Return chat model via Proxy (localhost:8000).

    Priority:
    1) Proxy if PROXY_URL is set
    3) Offline fallback
    """

    proxy_url = (os.getenv("PROXY_URL") or "").strip()
    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()

    # ---- 1) Use Proxy ----
    if proxy_url:
        try:
            from langchain_openai import ChatOpenAI
        except Exception as exc:
            raise RuntimeError("Install: pip install langchain-openai") from exc

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            api_key="proxy-key",  # dummy key (proxy ignores)
            base_url=f"{proxy_url}/v1",
        )
 

    # ---- 2) Offline fallback ----
    def _offline_response(prompt_value):
        try:
            messages = prompt_value.to_messages()
            last_user = next(
                (m.content for m in reversed(messages)
                 if getattr(m, "type", "") in ("human", "user")),
                "",
            )
        except Exception:
            last_user = ""

        content = (
            "[OFFLINE MODE] No Proxy/OpenAI configured. "
            f"You asked: {last_user}"
        )
        return AIMessage(content=content)

    return RunnableLambda(_offline_response)