import socket
import json
import os
import struct
from RunOptions import RunOptions
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ImageHandler(FileSystemEventHandler):
    def __init__(self, sock, img_dir):
        self.sock = sock
        self.img_dir = img_dir

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith((".png", ".jpg", ".jpeg", ".bmp", "JPEG")):
            send_image(self.sock, event.src_path)
            print(f"Sent image: {event.src_path}")
            predictions = receive_predictions(self.sock)
            print(f"Received predictions: {predictions}")

def send_image(sock, image_path):

    with open(image_path, "rb") as f:
        image_data = f.read()
    
    # Send the size of the image
    size = len(image_data)
    sock.sendall(struct.pack('!I', size))

    # Send the image data
    sock.sendall(image_data)


def receive_predictions(sock):
    # Receive the length of the JSON string
    size_data = receive_all(sock, struct.calcsize('!I'))
    size = struct.unpack('!I', size_data)[0]

    # Receive the JSON string
    predictions_json = receive_all(sock, size).decode('utf-8')

    # Deserialize JSON to Python object
    predictions = json.loads(predictions_json)
    return predictions

def receive_all(sock, size):
    data = b''
    while len(data) < size:
        part = sock.recv(size - len(data))
        if not part:
            break
        data += part
    return data

options = RunOptions.from_json("params.json")


if __name__ == "__main__":  
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("options:", options)
    print("options.target_image_paths:", options.target_image_paths)

    # Bind the socket to the address given on the command line
    server_address = ('localhost', 5001)
    print('starting up on {} port {}'.format(*server_address))
    sock.connect(server_address)
    img_dir = options.target_image_paths
    print("Sending images from directory:", img_dir)

    # Send all images in the directory
    image_paths = [image_path for image_path in os.listdir(img_dir) if image_path.endswith((".png", ".jpg", ".jpeg", ".bmp", "JPEG"))]
    for image_path in image_paths:
        full_image_path = os.path.join(img_dir, image_path)
        send_image(sock, full_image_path)
        predictions = receive_predictions(sock)
        print(f"Received predictions: {predictions}")

    # Set up the watchdog observer
    # Send new images as they are added to the directory
    event_handler = ImageHandler(sock, img_dir)
    observer = Observer()
    observer.schedule(event_handler, path=img_dir, recursive=False)
    observer.start()

    try:
        print("Monitoring directory for new images...")
        while True:
            pass  # Keep the script running
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

    print("closing socket")
    sock.close()