import socket
import selectors
import struct
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from collections import defaultdict, deque

parser = argparse.ArgumentParser()
parser.add_argument("--ports", nargs='+', type=int, default=[31200, 31201])
parser.add_argument("--log", type=str, help="CSV log output path")
args = parser.parse_args()

sel = selectors.DefaultSelector()
buffersize = 2048

sockets = {}
for port in args.ports:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', port))
    sock.setblocking(False)
    sel.register(sock, selectors.EVENT_READ)
    sockets[sock.fileno()] = port

last_recv_time = {}
delta_stats = defaultdict(lambda: {"min": float('inf'), "max": 0, "sum": 0, "count": 0})
delta_series = defaultdict(lambda: deque(maxlen=100))
tti_map = {}
timestamp_map = {}
log_file = open(args.log, 'w') if args.log else None

if log_file:
    log_file.write("port,timestamp_s,timestamp_ms,sample_offset,tti,crc,timestamp,delta_us\n")

fig, ax = plt.subplots()
lines = {}


def update_plot(frame):
    ax.clear()
    for port, series in delta_series.items():
        if series:
            xs, ys = zip(*series)
            line, = ax.plot(xs, ys, label=f"Port {port}")
            lines[port] = line
    ax.set_title("Δt (us) per Port")
    ax.set_ylabel("Microseconds")
    ax.set_xlabel("Sample Index")
    ax.legend()
    ax.grid(True)

ani = animation.FuncAnimation(fig, update_plot, interval=1000)

sample_idx = 0
while True:
    events = sel.select(timeout=0.01)
    now = time.monotonic()
    for key, _ in events:
        sock = key.fileobj
        port = sockets[sock.fileno()]
        data, addr = sock.recvfrom(buffersize)
        if len(data) < 12:
            continue

        header0 = data[0]
        debug_flag = bool(header0 & 0x20)
        null_flag = bool(header0 & 0x04)
        ms = ((data[0] & 0x03) << 8) | data[1]
        samples = (data[2] << 8) | data[3]
        s = struct.unpack_from("!I", data, 4)[0]

        offset = 8
        tti = crc = None
        if debug_flag and len(data) >= offset + 4:
            dbg_hdr = struct.unpack_from("!H", data, offset)[0]
            tti = dbg_hdr & 0x7FFF
            crc = struct.unpack_from("!H", data, offset + 2)[0]

            key = (s, ms)
            if key in timestamp_map and timestamp_map[key] != tti:
                print(f"[!] TTI mismatch for {s}.{ms}: seen {timestamp_map[key]}, now {tti}")
            timestamp_map[key] = tti

            if tti in tti_map and tti_map[tti] != crc:
                print(f"[!] CRC mismatch for TTI {tti}: was 0x{tti_map[tti]:04X}, now 0x{crc:04X}")
            tti_map[tti] = crc

        recv_ts = time.monotonic()
        if port in last_recv_time:
            delta_us = int((recv_ts - last_recv_time[port]) * 1e6)
            stats = delta_stats[port]
            stats["min"] = min(stats["min"], delta_us)
            stats["max"] = max(stats["max"], delta_us)
            stats["sum"] += delta_us
            stats["count"] += 1
            avg = stats["sum"] / stats["count"]
            delta_series[port].append((sample_idx, delta_us))

            print(f"Port {port} Δt = {delta_us} us, avg = {avg:.1f} us")

            if log_file:
                log_file.write(f"{port},{s},{ms},{samples},{tti},{crc},{recv_ts},{delta_us}\n")

        last_recv_time[port] = recv_ts
        sample_idx += 1

    plt.pause(0.001)

if log_file:
    log_file.close()
