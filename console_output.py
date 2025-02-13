import asyncio
import threading
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from scapy.all import sniff, IP, TCP, UDP

class AnomalyDetectionModel:
    def __init__(self, input_dim):
        self.model = Sequential([
            Input(shape=(input_dim,)),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    def train(self, x_train, y_train, epochs=10, batch_size=32):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    def predict(self, features):
        prediction = self.model.predict(np.array([features]), verbose=0)[0][0]
        return float(prediction)

class PacketClassificationModel:
    def __init__(self, input_dim, num_classes):
        self.model = Sequential([
            Input(shape=(input_dim,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        self.classes = ["Normal", "HTTP", "DNS", "ICMP", "Other"]
    
    def train(self, x_train, y_train, epochs=10, batch_size=32):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    def predict(self, features):
        predictions = self.model.predict(np.array([features]), verbose=0)[0]
        predicted_index = int(np.argmax(predictions))
        return self.classes[predicted_index], predictions.tolist()

def train_dummy_models():
    x_train = np.random.rand(1000, 7)
    y_train_anomaly = np.random.randint(0, 2, size=(1000, 1))
    y_train_class = tf.keras.utils.to_categorical(np.random.randint(0, 5, size=(1000, 1)), num_classes=5)
    anomaly_model = AnomalyDetectionModel(input_dim=7)
    classification_model = PacketClassificationModel(input_dim=7, num_classes=5)
    anomaly_model.train(x_train, y_train_anomaly, epochs=5)
    classification_model.train(x_train, y_train_class, epochs=5)
    return anomaly_model, classification_model

class FeatureExtractor:
    def extract_features(self, packet):
        features = []
        if packet.haslayer(IP):
            ip_layer = packet.getlayer(IP)
            pkt_len = float(len(packet))
            features.append(pkt_len)
            try:
                src_octet = float(int(ip_layer.src.split('.')[-1]))
            except Exception:
                src_octet = 0.0
            try:
                dst_octet = float(int(ip_layer.dst.split('.')[-1]))
            except Exception:
                dst_octet = 0.0
            features.extend([src_octet, dst_octet])
            proto = float(ip_layer.proto)
            features.append(proto)
            if ip_layer.proto == 6 and packet.haslayer(TCP):
                tcp_layer = packet.getlayer(TCP)
                features.append(float(tcp_layer.sport))
                features.append(float(tcp_layer.dport))
            elif ip_layer.proto == 17 and packet.haslayer(UDP):
                udp_layer = packet.getlayer(UDP)
                features.append(float(udp_layer.sport))
                features.append(float(udp_layer.dport))
            else:
                features.extend([0.0, 0.0])
        else:
            features = [0.0] * 7
        return np.array(features, dtype=float)

class PacketSniffer:
    def __init__(self, raw_packet_queue, interface, loop):
        self.raw_packet_queue = raw_packet_queue
        self.interface = interface
        self.loop = loop

    def packet_callback(self, packet):
        asyncio.run_coroutine_threadsafe(self.raw_packet_queue.put(packet), self.loop)

    def _sniff_packets(self):
        sniff(iface=self.interface, prn=self.packet_callback, store=False)

    def start(self):
        t = threading.Thread(target=self._sniff_packets, daemon=True)
        t.start()

async def feature_extraction_node(raw_packet_queue, anomaly_features_queue, classification_features_queue, extractor):
    event_counter = 0
    while True:
        packet = await raw_packet_queue.get()
        features = extractor.extract_features(packet)
        event = {"id": event_counter, "timestamp": time.time(), "features": features}
        await anomaly_features_queue.put(event)
        await classification_features_queue.put(event.copy())
        event_counter += 1

async def anomaly_detection_node(anomaly_features_queue, result_queue, anomaly_model):
    while True:
        event = await anomaly_features_queue.get()
        features = event["features"]
        score = anomaly_model.predict(features)
        result = {"id": event["id"], "timestamp": event["timestamp"], "anomaly_score": score, "type": "anomaly"}
        await result_queue.put(result)
        await asyncio.sleep(0.0001)

async def classification_node(classification_features_queue, result_queue, classification_model):
    while True:
        event = await classification_features_queue.get()
        features = event["features"]
        predicted_class, probabilities = classification_model.predict(features)
        result = {"id": event["id"], "timestamp": event["timestamp"], "predicted_class": predicted_class, "class_probabilities": probabilities, "type": "classification"}
        await result_queue.put(result)
        await asyncio.sleep(0.0001)

async def merger_node(result_queue):
    merged_events = {}
    while True:
        result = await result_queue.get()
        event_id = result["id"]
        if event_id not in merged_events:
            merged_events[event_id] = {}
        merged_events[event_id].update(result)
        if ("anomaly_score" in merged_events[event_id] and 
            "predicted_class" in merged_events[event_id]):
            combined_result = merged_events.pop(event_id)
            print("Merged Result:", combined_result)
        await asyncio.sleep(0.0001)

async def main():
    loop = asyncio.get_running_loop()
    raw_packet_queue = asyncio.Queue()
    anomaly_features_queue = asyncio.Queue()
    classification_features_queue = asyncio.Queue()
    result_queue = asyncio.Queue()

    extractor = FeatureExtractor()
    anomaly_model, classification_model = train_dummy_models()
    sniffer = PacketSniffer(raw_packet_queue, interface="eth0", loop=loop)
    sniffer.start()

    tasks = [
        asyncio.create_task(feature_extraction_node(raw_packet_queue, anomaly_features_queue, classification_features_queue, extractor)),
        asyncio.create_task(anomaly_detection_node(anomaly_features_queue, result_queue, anomaly_model)),
        asyncio.create_task(classification_node(classification_features_queue, result_queue, classification_model)),
        asyncio.create_task(merger_node(result_queue))
    ]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
