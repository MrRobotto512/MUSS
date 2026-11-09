"""Run this PC as a render node for MUSS (no window needed).

    python render_node.py [--port 47820] [--code PAIRINGCODE] [--gpus 0,1]

On the main PC: Settings -> Render nodes -> "+ Add node", then enter this PC's address and the
pairing code printed below. Allow Python through Windows Firewall (Private networks) when asked.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from upscaler import node  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=node.DEFAULT_PORT)
    ap.add_argument("--code", default=None, help="pairing code (default: a new random code)")
    ap.add_argument("--gpus", default=None, help="comma-separated GPU indexes to use (default: all)")
    a = ap.parse_args()
    gpus = [int(x) for x in a.gpus.split(",")] if a.gpus else None
    srv = node.NodeServer(port=a.port, code=a.code, gpu_ids=gpus,
                          log=lambda m: print(time.strftime("%H:%M:%S ") + m, flush=True)).start()
    print(f"\n  Address:      {node.lan_ip()}:{a.port}\n  Pairing code: {srv.code}\n\nPress Ctrl+C to stop.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        srv.stop()


if __name__ == "__main__":
    main()
