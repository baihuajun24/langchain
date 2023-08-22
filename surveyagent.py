from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

def read_key_from_file(filename):
    with open(filename, 'r') as file:
        key = file.readline().strip()  # read the first line and remove newline character
    return key

key_file = 'key.txt'  # Replace with the actual path to your text file
api_key = read_key_from_file(key_file)
chat = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0.3, openai_api_key=api_key)
sys_msg = SystemMessage(content="Your employer Kibra which sells sports bras wants to do NPS (net promoter score) survey and prompt follow-up open-ended questions. \
                        Works as a survey agent and converse with the customer.\
                        You should start by asking the customer this question: \
                         How likely is it that you would recommend Kibra to a friend or colleague? Please pick a score between 0 and 10. 0 means least likely, while 10 means extremely likely")
# messages = [
#     SystemMessage(content="Your employer Kibra which sells sports bras wants to do NPS (net promoter score) survey and prompt follow-up open-ended questions. Works as a survey agent and converse with the customer."),
#     HumanMessage(content="You should start by asking the customer this question: \
#                  How likely is it that you would recommend Kibra to a friend or colleague? Please pick a score between 0 and 10. 0 means least likely, while 10 means extremely likely")
# ]
# response=chat(messages)

# print(response.content,end='\n')

import socket

# Define the host and port
HOST = 'localhost'
PORT = 12345

# Create the socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the host and port
server_socket.bind((HOST, PORT))

# Listen for incoming connections
server_socket.listen()

# Accept the client connection
client_socket, client_address = server_socket.accept()
print('Connected to:', client_address)

# Loop for chatting round after round
while True:
    # Receive message from the client
    client_message = client_socket.recv(1024).decode('utf-8')
    print('Client:', client_message)

    # Process the client message and generate server response
    # Replace the following line with your code
    messages = [
        sys_msg,
        HumanMessage(content=client_message)
    ]
    server_response = chat(messages).content

    # Send the server response to the client
    client_socket.send(server_response.encode('utf-8'))

    # Save the chat history on the fly
    with open('chat_history.txt', 'a') as file:
        file.write('Client: ' + client_message + '\n')
        file.write('Server: ' + server_response + '\n')

# Close the server socket
server_socket.close()