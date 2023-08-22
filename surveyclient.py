import socket

# Define the host and port
HOST = 'localhost'
PORT = 12345

# Create the socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the server
client_socket.connect((HOST, PORT))
print('Connected to the server.')

# Loop for chatting round after round
while True:
    # Get user input for message
    user_input = input('You: ')

    # Send the user input to the server
    client_socket.send(user_input.encode('utf-8'))

    # Receive the server response
    server_response = client_socket.recv(1024).decode('utf-8')
    print('Server:', server_response)

    # Exit the loop if user input is 'bye'
    if user_input.lower() == 'bye':
        break

# Close the client socket
client_socket.close()