import socket
import numpy as np
import sys
import signal
from RunOptions import RunOptions
from tensorboardX import SummaryWriter
import torch
import json
import os
import struct
from osod import zero_shot_detection, one_shot_detection_batches, find_query_patches_batches, visualize_results, add_query
from config import query_dir, test_dir, results_dir
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from threading import Lock

class ImageHandler(FileSystemEventHandler):
    def __init__(self, model, processor, options, query_embeddings, classes, writer, cls, lock):
        self.model = model
        self.processor = processor
        self.options = options
        self.query_embeddings = query_embeddings
        self.classes = classes
        self.writer = writer
        self.cls = cls
        self.lock = lock

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith((".png", ".jpg", ".jpeg", ".bmp", "JPEG")):
            print(f"New query image detected: {event.src_path}")
            with self.lock:
                self.query_embeddings, self.classes = add_query(
                    self.model,
                    self.processor,
                    self.options,
                    event.src_path,
                    self.query_embeddings,
                    self.classes,
                    self.writer,
                    self.cls
                )
                print("Updated query_embeddings and classes.")


def receive_all(connection, size):
    data = b''
    while len(data) < size:
        part = connection.recv(size - len(data))
        if not part:
            break
        data += part
    return data

def save_image(data, dir, filename, dtype, dims):
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    # Save the image to the specified directory
    filepath = os.path.join(dir, filename)
    data = np.frombuffer(data, dtype=dtype).reshape(dims[0], dims[1], dims[2])

    from matplotlib import pyplot as plt
    plt.imsave(filepath, data)

    #with open(filepath, 'wb') as f:
        # f.write(data)
    #    pass
    print("Saving image to filepath: ", filepath)
    return filepath

def signal_handler(sig, frame):
    print('Shutting down server...')
    observer.stop()  # Stop the watchdog observer
    observer.join()  # Wait for the observer thread to finish
    sock.close()     # Close the socket only after stopping the observer
    sys.exit(0)

def zero_shot():
    # Find the objects in the query images
    cls = [1]*12
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
            writer,
            cls
        )

    with open(f"Queries\\classes_{options.comment}.json", 'w') as f:
        json.dump(classes, f)

    # Save the list of GPU tensors to a file
    torch.save(query_embeddings, f"Queries\\query_embeddings_{options.comment}_gpu.pth")

    return query_embeddings, classes

def send_predictions(connection, predictions):
   
    # Send the length of the JSON string
    size = len(predictions)
    connection.sendall(str(size).encode('utf-8'))
    print(connection.recv(1024).decode())
    
    # Send the JSON string
    connection.sendall(predictions.encode('utf-8'))
    print(connection.recv(1024).decode())


def find_grasping_points(data, predictions):
    # Get the grasping points for each object
    grasping_points = []
    for i, pred in enumerate(predictions):
        # Get the bounding box for the object
        bbox = pred['bbox']
        category = pred['category_id']
        x, y, w, h = bbox
        
        # compute centroid
        cx, cy = int(x + w/2), int(y + h/2)

        # find xyz and nx ny nz
        grasping_point = [d[cy,cx,0] for d in data]
        grasping_point.append(category)
        grasping_points.append(grasping_point)

    # sort by increasing value of z
    grasping_points = sorted(grasping_points, key=lambda x: x[2])
    grasping_points = np.array(grasping_points)
    grasping_points = np.array2string(grasping_points, separator=' ', precision=3)
    
    return grasping_points  

# Lock for shared resource protection
lock = Lock()
dir = "received_images/"
if not os.path.exists(dir):
    os.makedirs(dir)
    
options = RunOptions.from_json("params.json")
options.target_image_paths = dir
writer = SummaryWriter(comment=options.comment)
model = options.model.from_pretrained(options.backbone)
processor = options.processor.from_pretrained(options.backbone)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

query_directory = options.source_image_paths


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
    


    file = os.path.join(query_dir, f"classes_{options.comment}.json")
    with open(file, 'r') as f:
        classes = json.load(f)
    

    # Load the list of tensors onto the GPU
  #  query_embeddings = torch.load(f'Queries/query_embeddings_{options.comment}_gpu.pth', map_location=device)
    
    
    query_embeddings, classes = zero_shot()
    cls = [0]

    # Create an observer to watch for new query images
    event_handler = ImageHandler(model, processor, options, query_embeddings, classes, writer, cls, lock)
    observer = Observer()
    observer.schedule(event_handler, path=query_directory, recursive=False)
    observer.start()

    file = os.path.join(query_dir, f"classes_{options.comment}.json")
    with open(file, 'r') as f:
        classes = json.load(f)

    # Load the list of tensors onto the GPU
    query_embeddings = torch.load(f'Queries/query_embeddings_{options.comment}_gpu.pth', map_location=device)
    
    initial_batch_processed = False

    try:    
        print("Initializing the connection :D")
        while True: 
            
            try:
                
                while True:
                    print('waiting for a connection')
                    connection, client_address = sock.accept()
                    print('connection from', client_address)

                    # Receive the size of the image
                    # size_data = receive_all(connection, 1 )
                    # if not size_data:
                    #     break
                    # w = struct.unpack('!I', size_data)[0]
                    w = int(connection.recv(1024).decode())
                    connection.sendall(b'w')
                    print("Image width", w)

                    # size_data = receive_all(connection, struct.calcsize('!I'))
                    # if not size_data:
                    #     break
                    #h = struct.unpack('!I', size_data)[0]
                    h = int(connection.recv(1024).decode())
                    connection.sendall(b'h')
                    print("Image height", h)

                    # Receive the image data
                    image_data = receive_all(connection, w*h*3) # 3 channels for RGB
                    if not image_data:
                        break
                    connection.sendall(b'rgb')


                    data = []
                    for i in range(6):
                        d =  receive_all(connection, w*h*4) # 4 bytes for each float
                        if not d:
                            break
                        d = np.frombuffer(d, dtype=np.float32)
                        d = d.reshape(h,w,1)
                        data.append(d)
                        connection.sendall(b'float')
                        
                    # Save the image
                    filename = f'test_image.jpg'
                    path = save_image(image_data, dir, filename, dtype=np.uint8, dims=(h, w, 3))

                    options.target_image_paths = path   # the path used in osod.py visualize_results

                    with lock:  
                        current_query_embeddings = query_embeddings
                        current_classes = classes

                    # Perform one-shot detection for new images
                
                    id, predictions= one_shot_detection_batches(
                        model=model,
                        processor=processor,
                        args=options,
                        writer=writer,
                        query_embeddings = current_query_embeddings,
                        classes = current_classes,
                        per_image=True
                    )

                    grasping_points = find_grasping_points(data, predictions)
                    print(grasping_points)
                    # Send predictions back to the client
                    send_predictions(connection, grasping_points)
                    writer.add_text("Predictions", json.dumps(predictions))
                    if options.visualize_test_images:
                        filepath = os.path.join(results_dir, f"results_{id}.json")
                        visualize_results(filepath, writer, per_image=True, args=options, random_selection=None)
                    
                    print(f'Sent predictions for {filename}')


            except socket.timeout:
                continue
            except KeyboardInterrupt:
                print('Shutting down server...')
                break
    finally:
        observer.stop()
        observer.join()
        sock.close()
        sys.exit(0)
        # Clean up the connection
        if 'connection' in locals():
            connection.close()