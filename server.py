import socket
import sys
import signal
from RunOptions import RunOptions
from tensorboardX import SummaryWriter
import torch
import json
import os
import struct
from osod import zero_shot_detection, one_shot_detection_batches, find_query_patches_batches, visualize_results
from config import query_dir, test_dir, results_dir

def receive_all(connection, size):
    data = b''
    while len(data) < size:
        part = connection.recv(size - len(data))
        if not part:
            break
        data += part
    return data

def save_image(data, dir, filename):
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    # Save the image to the specified directory
    filepath = os.path.join(dir, filename)
    with open(filepath, 'wb') as f:
        f.write(data)
    print("Saving image to filepath: ", filepath)
    return filepath

def signal_handler(sig, frame):
    print('Shutting down server...')
    sock.close()
    sys.exit(0)

def zero_shot():
     # Find the objects in the query images
    if options.manual_query_selection:
        zero_shot_detection(model, processor, options, writer)
        #indexes = [1523, 1700, 1465, 1344]
        indexes = [1523, 1641, 1750, 1700, 1700, 747, 1465, 1704, 1214, 1344, 876, 2071]
        query_embeddings, classes = find_query_patches_batches(
            model, processor, options, indexes, writer
        )

    else:
        indexes, query_embeddings, classes = zero_shot_detection(
            model,
            processor,
            options,
            writer
        )

    with open("classes.json", 'w') as f:
        json.dump(classes, f)

    # Save the list of GPU tensors to a file
    torch.save(query_embeddings, 'query_embeddings_gpu.pth')

def send_predictions(connection, predictions):
    # Serialize predictions to JSON
    predictions_json = json.dumps(predictions)
    
    # Send the length of the JSON string
    size = len(predictions_json)
    connection.sendall(struct.pack('!I', size))
    
    # Send the JSON string
    connection.sendall(predictions_json.encode('utf-8'))


dir = "received_images/"
if not os.path.exists(dir):
    os.makedirs(dir)
    
options = RunOptions.from_json("params.json")
options.target_image_paths = dir
writer = SummaryWriter(comment=options.comment)
model = options.model.from_pretrained(options.backbone)
processor = options.processor.from_pretrained(options.backbone)


if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal_handler)
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the address given on the command line
    server_address = ('localhost', 5001)
    print('starting up on {} port {}'.format(*server_address))
    sock.bind(server_address)

    # Listen for incoming connections
    sock.listen(1)

    # Set the socket to non-blocking mode
    sock.settimeout(5)
    
    #zero_shot()

    
    file = os.path.join(query_dir, f"classes_{options.data}.json")
    with open(file, 'r') as f:
        classes = json.load(f)

    # Load the list of tensors onto the GPU
    query_embeddings = torch.load(f'Queries/query_embeddings_{options.data}_gpu.pth', map_location='cuda')
    

    counter = 0
    initial_batch_processed = False

    while True: 
        try:
            print('waiting for a connection')
            connection, client_address = sock.accept()
            print('connection from', client_address)
        

            while True:
                # Receive the size of the image
                size_data = receive_all(connection, struct.calcsize('!I'))
                if not size_data:
                    break
                size = struct.unpack('!I', size_data)[0]

                # Receive the image data
                image_data = receive_all(connection, size)
                if not image_data:
                    break

                # Save the image
                filename = f'img_{counter}.jpg'
                path = save_image(image_data, dir, filename)
                counter += 1

                options.target_image_paths = path
                
                # Perform one-shot detection for new images
                id, predictions= one_shot_detection_batches(
                    model,
                    processor,
                    query_embeddings,
                    classes,
                    options,
                    writer,
                    per_image=True
                )

                
                # Send predictions back to the client
                send_predictions(connection, predictions)

                if options.visualize_test_images:
                    filepath = os.path.join(results_dir, f"results_{id}.json")
                    visualize_results(filepath, writer, per_image=True, args=options, random_selection=None)
                
              
                print(f'Sent predictions for {filename}')


        except socket.timeout:
            continue
        except KeyboardInterrupt:
            print('Shutting down server...')
            sock.close()
            sys.exit(0)
        finally:
            # Clean up the connection
            if 'connection' in locals():
                connection.close()