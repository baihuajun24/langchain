from langchain import FewShotPromptTemplate, PromptTemplate
from langchain.chains.sql_database.prompt import _sqlite_prompt, PROMPT_SUFFIX
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts.example_selector.semantic_similarity import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from langchain.llms.glm import GLM
import yaml
from flask import Flask, request, jsonify
import argparse
from typing import Dict

def read_key_from_file(filename):
    with open(filename, 'r') as file:
        key = file.readline().strip()  # read the first line and remove newline character
    return key

def _parse_example(result: Dict) -> Dict:
    sql_cmd_key = "sql_cmd"
    sql_result_key = "sql_result"
    table_info_key = "table_info"
    input_key = "input"
    final_answer_key = "answer"

    _example = {
        "input": result.get("query"),
    }

    steps = result.get("intermediate_steps")
    answer_key = sql_cmd_key # the first one
    for step in steps:
        # The steps are in pairs, a dict (input) followed by a string (output).
        # Unfortunately there is no schema but you can look at the input key of the
        # dict to see what the output is supposed to be
        if isinstance(step, dict):
            # Grab the table info from input dicts in the intermediate steps once
            if table_info_key not in _example:
                _example[table_info_key] = step.get(table_info_key)

            if input_key in step:
                if step[input_key].endswith("SQLQuery:"):
                    answer_key = sql_cmd_key # this is the SQL generation input
                if step[input_key].endswith("Answer:"):
                    answer_key = final_answer_key # this is the final answer input
            elif sql_cmd_key in step:
                _example[sql_cmd_key] = step[sql_cmd_key]
                answer_key = sql_result_key # this is SQL execution input
        elif isinstance(step, str):
            # The preceding element should have set the answer_key
            _example[answer_key] = step
    return _example

YAML_EXAMPLES = """
- input: 今天是2023-06-28，昨天互动量最高的微博账号是?
  table_info: |
    CREATE TABLE social_reaction (
        the_date Date NOT NULL,
        type TEXT NOT NULL,
        name TEXT NOT NULL,
        content TEXT NOT NULL,
        url TEXT NOT NULL,
        reaction INTEGER NOT NULL,
        language TEXT,
        programe TEXT,
        plate_name TEXT
    )
  sql_cmd: 
    SELECT the_date, name, sum(reaction) as total_react
    FROM social_reaction
    WHERE type = '微博' and name is not 'n/a' and the_date = '2023-06-27'
    group by the_date, name
    ORDER BY total_react DESC
    LIMIT 1;
  sql_result: "2023-06-27|央视新闻|3098"
  answer: 昨天2023年6月27日，互动量最高的微博账号是央视新闻, 它的互动量是3098。
- input: 2023-06-27这一天互动量最高的微信账号是?
  table_info: |
    CREATE TABLE social_reaction (
        the_date Date NOT NULL,
        type TEXT NOT NULL,
        name TEXT NOT NULL,
        content TEXT NOT NULL,
        url TEXT NOT NULL,
        reaction INTEGER NOT NULL,
        language TEXT,
        programe TEXT,
        plate_name TEXT
    )
  sql_cmd: 
    SELECT the_date, name, sum(reaction) as total_react
    FROM social_reaction
    WHERE type = '微信' and name is not 'n/a' and the_date = '2023-06-27'
    group by the_date, name
    ORDER BY total_react DESC
    LIMIT 1;
  sql_result: "2023-06-27|央视一套|9690"
  answer: 2023年6月27日，互动量最高的微信账号是央视一套, 它的互动量是9690。
- input: 2023-06-27这一天总台互动量最低的微博账号是?
  table_info: |
    CREATE TABLE social_reaction (
        the_date Date NOT NULL,
        type TEXT NOT NULL,
        name TEXT NOT NULL,
        content TEXT NOT NULL,
        url TEXT NOT NULL,
        reaction INTEGER NOT NULL,
        language TEXT,
        programe TEXT,
        plate_name TEXT
    )
  sql_cmd: 
    SELECT the_date, name, sum(reaction) as total_react
    FROM social_reaction
    WHERE type = '微博' 
      and name is not 'n/a' 
      and the_date = '2023-06-27'
    group by the_date, name
    HAVING total_react = 0;
  sql_result: "2023-06-27|1012交通广播|0
2023-06-27|CCTV-17地球村日记|0
2023-06-27|CCTV农业气象|0"
  answer: 2023年6月27日，互动量最低的微博账号包括1012交通广播、CCTV-17地球村日记、CCTV农业气象，互动量都为0。
- input: 2023-06-27这一天互动量最高的微信账号是?
  table_info: |
    CREATE TABLE social_reaction (
        the_date Date NOT NULL,
        type TEXT NOT NULL,
        name TEXT NOT NULL,
        content TEXT NOT NULL,
        url TEXT NOT NULL,
        reaction INTEGER NOT NULL,
        language TEXT,
        programe TEXT,
        plate_name TEXT
    )
  sql_cmd: 
    SELECT the_date, name, sum(reaction) as total_react
    FROM social_reaction
    WHERE type = '微信' and name is not 'n/a' and the_date = '2023-06-27'
    group by the_date, name
    ORDER BY total_react DESC
    LIMIT 1;
  sql_result: "2023-06-27|央视一套|9690"
  answer: 2023年6月27日，互动量最高的微信账号是央视一套, 它的互动量是9690。
- input: 今天是2023年7月5日，本周总台互动量最高的微博账号是？
  table_info: |
    CREATE TABLE social_reaction (
        the_date Date NOT NULL,
        type TEXT NOT NULL,
        name TEXT NOT NULL,
        content TEXT NOT NULL,
        url TEXT NOT NULL,
        reaction INTEGER NOT NULL,
        language TEXT,
        programe TEXT,
        plate_name TEXT
    )
  sql_cmd: 
    SELECT name, sum(reaction) as total_reaction
    FROM social_reaction
    WHERE type = '微博' 
      and name is not 'n/a' 
      and the_date >= date('2023-07-05', '-7 days')
    GROUP BY name
    ORDER BY total_reaction DESC
    LIMIT 1;
  sql_result: "央视新闻|33448"
  answer: 2023年7月5日以前的一周，互动量最高的微博账号是央视新闻，互动量为33448。
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
                        k=min(2, len(examples_dict)),
                    )

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=_sqlite_prompt + "Don't use \_ in sql. Here are some examples:",
    suffix=PROMPT_SUFFIX,
    input_variables=["table_info", "input", "top_k"],
)

my_table_info = {
    "social_reaction": """CREATE TABLE social_reaction (
        the_date Date NOT NULL,
        type TEXT NOT NULL,
        name TEXT NOT NULL,
        content TEXT NOT NULL,
        url TEXT NOT NULL,
        reaction INTEGER NOT NULL,
        language TEXT,
        programe TEXT,
        plate_name TEXT
    )
)
"""
}

db = SQLDatabase.from_uri("sqlite:///D:/huajun/softwares/litestream-0.3.9/data_0706.db", include_tables=['social_reaction'], custom_table_info=my_table_info)
key_file = 'key.txt'  # Replace with the actual path to your text file
api_key = read_key_from_file(key_file)
local_llm = OpenAI(temperature=0, openai_api_key=api_key)
# local_llm = GLM()
# local_llm.load_model(model_name_or_path='D:\\huajun\\chatGLM\\chatGLM\\chatglm2-6b')
# local_llm = OpenLLM(model_name="chatglm", model_id='D:\\huajun\\chatGLM\\chatGLM\\chatglm2-6b')
# local_llm("What is the difference between a duck and a goose? And why there are so many Goose in Canada?")
# local_chain = SQLDatabaseChain.from_llm(local_llm, db, prompt=few_shot_prompt, use_query_checker=True, verbose=True, return_intermediate_steps=True)
# Test without checker for local llm
local_chain = SQLDatabaseChain.from_llm(local_llm, db, prompt=few_shot_prompt, verbose=True, return_intermediate_steps=True)

# result = local_chain("今天是2023-06-28，昨天总台互动量最高的微博账号是?") # success, for local llm
# result = local_chain("昨天总台互动量最高的微博账号是?") # not success, the_date = DATE('now', '-1 day')
# result = local_chain("今天是2023-06-28，昨天总台互动量最高的微信账号是?") # not sucess, name IS NOT NULL; glm2 asks on table _litestream_seq
# result = local_chain("2023-06-28这一天互动量最高的微博账号是?") # success
# result = local_chain("2023-06-27这一天互动量最高的微博账号是?") # success
# result = local_chain("2023-06-27这一天互动量最高的微信账号是?") # not sucess, name IS NOT NULL; glm2 SQLResult: 2023-06-27|央视财经|4112, 但是返回报错
# result = local_chain("2023-06-28这一天互动量最高的微信账号是?") # success
# result = local_chain("2023-06-28这一天总台互动量最低的微博账号是？")
# print(result)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--server', action='store_true')
args = parser.parse_args()

def run_tests():
    # Put your testing code here. For example:
    queries = [
        "2023-06-28总台互动量最高的微博账号是?",
        "2023-06-28这一天总台互动量最低的微博账号是？",
        #... add more queries for testing
    ]
    result = ''
    for query in queries:
        try:
          result += local_chain(query)
        except:
          result += "Error occurred\n"
        print(result)

if __name__ == '__main__':
    if args.server:
        # Start server
        app = Flask(__name__)

        @app.route('/query', methods=['POST'])
        def process_query():
            try:
                data = request.get_json()
                query = data['query']
                result = local_chain(query)
                print("*** Query succeeded")
                example = _parse_example(result)
                return jsonify({'result': str(result)})
            except Exception as e:
                  print("*** Query failed")
                  result = {
                      "query": query,
                      "intermediate_steps": e.intermediate_steps
                  }
                  example = _parse_example(result)
                  print(f"Error occurred: {str(e)}")  # Log the error
                  return jsonify({'error': 'An error occurred processing the request.'}), 500
            yaml_example = yaml.dump(example, allow_unicode=True)
            print("\n" + yaml_example)
        app.run(host='0.0.0.0', port=5000)
    else:
        # Run tests
        run_tests()