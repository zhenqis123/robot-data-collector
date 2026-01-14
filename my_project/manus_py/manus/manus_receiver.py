import redis
import pickle
import numpy as np
import time

def main():
    # Connect to Redis
    r = redis.Redis(host='localhost', port=6379, db=0)
    
    print("Listening for Manus glove data on Redis key 'manus'...")
    
    try:
        while True:
            # Get data from Redis
            data_bytes = r.get('manus')
            
            if data_bytes:
                try:
                    # Unpickle the data
                    data = pickle.loads(data_bytes)
                    
                    # Print received keys and shapes
                    print(f"\nReceived data timestamp: {time.time():.4f}")
                    for key, value in data.items():
                        if isinstance(value, np.ndarray):
                            print(f"  {key}: shape {value.shape}")
                        else:
                            print(f"  {key}: {type(value)}")
                    
                    '''
                    Received data timestamp: 1768370929.8588
                    right_fingers: shape (25, 3)
                    right_orientations: shape (25, 4)
                    left_fingers: shape (25, 3)
                    left_orientations: shape (25, 4)
                    '''
                except pickle.UnpicklingError:
                    print("Error unpickling data")
            else:
                print("No data in 'manus' key yet...")
            
            time.sleep(0.1) # 10Hz print rate to avoid collecting too much log
            
    except KeyboardInterrupt:
        print("\nStopped listener.")

if __name__ == "__main__":
    main()
