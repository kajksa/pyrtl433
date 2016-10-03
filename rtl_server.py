import argparse
import rtlsdr


#hostname = "127.0.0.1"
hostname = "192.168.0.2"
port = 1235
device_index = 0

p = argparse.ArgumentParser()
p.add_argument(
    '-a', '--address',
    dest='hostname',
    metavar='address',
    default='127.0.0.1',
    help='Listen address (default is "127.0.0.1")')
p.add_argument(
    '-p', '--port',
    dest='port',
    type=int,
    default=1235,
    help='Port to listen on (default is 1235)')
p.add_argument(
    '-d', '--device-index',
    dest='device_index',
    type=int,
    default=0)


args, remaining = p.parse_known_args()


print("Starting rtlsdr server on {} port {}".format(args.hostname, args.port))

server = rtlsdr.RtlSdrTcpServer(hostname=args.hostname, port=args.port, device_index=args.device_index)
server.run_forever()
