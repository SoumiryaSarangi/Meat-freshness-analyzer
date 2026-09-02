"""
Routing layer: turns a pipeline decision ("discard", "grinding", "packing")
into a signal sent to actual plant hardware. This is the one module you will
need to adapt to your specific setup (PLC, relay board, conveyor diverter,
etc) -- everything above this layer is hardware-agnostic.

Three backends are stubbed out:
  - console: just logs the decision (default, safe for development)
  - serial:  sends a single-byte command over a serial line to a
             microcontroller (Arduino/ESP32) driving relays or diverters
  - mqtt:    publishes the decision to an MQTT topic, for PLCs/SCADA
             systems that subscribe over a message broker

Fill in the TODOs with your actual wiring/protocol.
"""

import csv
import os
import time
from abc import ABC, abstractmethod


class Router(ABC):
    @abstractmethod
    def route(self, decision: str, piece_id: int, meta: dict) -> None:
        """decision is one of: 'discard', 'grinding', 'packing'."""
        raise NotImplementedError


class ConsoleRouter(Router):
    """Default backend: logs to stdout and a CSV file. No hardware required --
    use this to validate detection/classification logic before wiring up
    actuators."""

    def __init__(self, log_csv_path: str = None):
        self.log_csv_path = log_csv_path
        if log_csv_path:
            os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
            if not os.path.exists(log_csv_path):
                with open(log_csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", "piece_id", "decision",
                                      "good_confidence", "spoiled_confidence",
                                      "size_category", "longest_dimension_mm"])

    def route(self, decision: str, piece_id: int, meta: dict) -> None:
        print(f"[ROUTE] piece={piece_id} -> {decision.upper()} | {meta}")
        if self.log_csv_path:
            with open(self.log_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    piece_id,
                    decision,
                    meta.get("good_confidence", ""),
                    meta.get("spoiled_confidence", ""),
                    meta.get("size_category", ""),
                    meta.get("longest_dimension_mm", ""),
                ])


class SerialRouter(Router):
    """Sends a single command byte per decision to a microcontroller that
    drives the physical diverter/discard mechanism. Adjust the command map
    and baud rate to match your firmware."""

    COMMAND_MAP = {"discard": b"D", "grinding": b"G", "packing": b"P"}

    def __init__(self, port: str, baud: int = 9600):
        import serial  # local import: optional dependency
        self.conn = serial.Serial(port, baud, timeout=1)

    def route(self, decision: str, piece_id: int, meta: dict) -> None:
        command = self.COMMAND_MAP.get(decision)
        if command is None:
            raise ValueError(f"Unknown decision: {decision}")
        # TODO: adjust framing to match your microcontroller's protocol
        # (e.g. include piece_id, a checksum, or a line terminator).
        self.conn.write(command)


class MQTTRouter(Router):
    """Publishes routing decisions to an MQTT topic for PLC/SCADA systems
    that subscribe over a broker rather than direct serial/GPIO."""

    def __init__(self, broker: str, topic: str, port: int = 1883):
        import paho.mqtt.client as mqtt  # local import: optional dependency
        import json
        self._json = json
        self.topic = topic
        self.client = mqtt.Client()
        self.client.connect(broker, port)
        self.client.loop_start()

    def route(self, decision: str, piece_id: int, meta: dict) -> None:
        payload = self._json.dumps({"piece_id": piece_id, "decision": decision, **meta})
        self.client.publish(self.topic, payload)


def build_router(config: dict) -> Router:
    backend = config["routing"]["backend"]
    if backend == "console":
        return ConsoleRouter(log_csv_path=config.get("output", {}).get("log_csv"))
    if backend == "serial":
        return SerialRouter(config["routing"]["serial_port"], config["routing"]["serial_baud"])
    if backend == "mqtt":
        return MQTTRouter(config["routing"]["mqtt_broker"], config["routing"]["mqtt_topic"])
    raise ValueError(f"Unknown routing backend: {backend}")
