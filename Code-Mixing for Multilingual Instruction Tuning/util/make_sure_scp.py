from util.ssh_add_server import scp, gsutil_cp
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('src')
    parser.add_argument('dst')
    parser.add_argument('--server', default='root@dilab4.snu.ac.kr', type=str, help="server, like root@a.b.kr")
    parser.add_argument('-p', '--port', type=int, default=11001)
    parser.add_argument('--rsa_key', type=str, default='~/.ssh/dilab4')
    args = parser.parse_args()

    scp(args.src, args.dst, args.server, args.port, args.rsa_key)