from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.llms.glm import GLM
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

db = SQLDatabase.from_uri("sqlite:///D:/huajun/softwares/litestream-0.3.9/data_backup.db")
# llm = OpenAI(temperature=0, openai_api_key='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')
llm = GLM()
llm.load_model(model_name_or_path='D:\\huajun\\chatGLM\\chatGLM\\chatglm-6B')
# llm.load_model(model_name_or_path='D:\\huajun\\chatGLM\\chatGLM\\chatglm2-6b') # V2
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

agent_executor.run("Describe the table social_reaction")
# agent_executor.run("互动量最高的微博账号是？") # 昨天互动量最高的微博账号是?
# agent_executor.run("which type has highest reactions in table social_reaction？")
# agent_executor.run("which type='微博' name has highest reaction？")
