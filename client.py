import requests
import json
import ast

# Base URL of the server
url = "http://127.0.0.1:5000/query"

while True:
    # Take user input
    query = input("Enter your query: ")

    # Create JSON payload
    data = {'query': query}

    # Send POST request to server
    response = requests.post(url, json=data)
    # Send the POST request to the server
    # response = requests.post(url, json={"query": query})

    # If the request was successful, decode the JSON response
    if response.status_code == 200:
        result = response.json()
        final_result = result['result']
        print("Result: ", final_result)
        
        # result_dict = ast.literal_eval(result['result'])
        # final_result = result_dict['result']
        # print("Result: ", final_result)
        # print("Result: ", result["result"])
    else:
        print(f"Request failed with status code {response.status_code}")