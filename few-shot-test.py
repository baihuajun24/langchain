from langchain import FewShotPromptTemplate, PromptTemplate
from langchain.chains.sql_database.prompt import _sqlite_prompt, PROMPT_SUFFIX
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts.example_selector.semantic_similarity import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
import yaml

YAML_EXAMPLES = """
- input: 互动量最高的微博账号是?
  table_info: |
    CREATE TABLE social_reaction (
        type TEXT NOT NULL, 
        name TEXT NOT NULL, 
        content TEXT NOT NULL, 
        url TEXT NOT NULL, 
        reaction INTEGER
    )
  sql_cmd: 
    SELECT name, sum(reaction) as total_react
    FROM social_reaction
    WHERE type = '微博' and name is not 'n/a'
    group by name
    ORDER BY total_react DESC
    LIMIT 1;
  sql_result: "CCTV电视剧|1369"
  answer: 互动量最高的微博账号是CCTV电视剧, 总互动量是1369.
"""

example_prompt = PromptTemplate(
    input_variables=["table_info", "input", "sql_cmd", "sql_result", "answer"],
    template="{table_info}\n\nQuestion: {input}\nSQLQuery: {sql_cmd}\nSQLResult: {sql_result}\nAnswer: {answer}",
)

examples_dict = yaml.safe_load(YAML_EXAMPLES)

local_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

example_selector = SemanticSimilarityExampleSelector.from_examples(
                        # This is the list of examples available to select from.
                        examples_dict,
                        # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
                        local_embeddings,
                        # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
                        Chroma,  # type: ignore
                        # This is the number of examples to produce and include per prompt
                        k=min(3, len(examples_dict)),
                    )

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=_sqlite_prompt + "Here are some examples:",
    suffix=PROMPT_SUFFIX,
    input_variables=["table_info", "input", "top_k"],
)

db = SQLDatabase.from_uri("sqlite:///D:/huajun/softwares/litestream-0.3.9/data_backup.db")
local_llm = OpenAI(temperature=0, openai_api_key='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')
local_chain = SQLDatabaseChain.from_llm(local_llm, db, prompt=few_shot_prompt, use_query_checker=True, verbose=True, return_intermediate_steps=True)

result = local_chain("互动量前五的微博账号是哪些?")

print(result)