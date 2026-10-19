from langchain_openai import ChatOpenAI

LLM_SEED = 42

llm = ChatOpenAI(
    model="gpt-5",
    temperature=0,
    seed=LLM_SEED,
)
