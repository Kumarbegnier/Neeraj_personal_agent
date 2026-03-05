from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
import logging

logger = logging.getLogger(__name__)

def load_prompt(file_path: str) -> str:

    try :
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        logger.exception(f"Failed loading prompt: {file_path}")
        raise RuntimeError(f"Prompt unavailable: {os.path.basename(file_path)}")



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


system_prompt = load_prompt(os.path.join(BASE_DIR, "prompts", "system.txt"))
user_prompt = load_prompt(os.path.join(BASE_DIR, "prompts", "user.txt"))


chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("history"),
    ("human", user_prompt)
])


# # Backwards-compatible: if a prompt file still references {profile},
# # provide an empty default so the API doesn't fail.
# try:
#     if "profile" in set(getattr(chat_prompt, "input_variables", []) or []):
#         chat_prompt = chat_prompt.partial(profile="")
# except Exception:
#     logger.exception("Failed applying prompt partial defaults")

# try:
#     logger.info("Prompt variables: %s", getattr(chat_prompt, "input_variables", None))
# except Exception:
#     pass
