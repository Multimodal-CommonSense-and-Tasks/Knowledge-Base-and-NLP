import argparse, subprocess, os
error_str = 'Connection reset by peer'
DEBUG = True

max_attempts = 99999999
def make_sure_run(run_script, debug=DEBUG, attempts=max_attempts):
    for _ in range(attempts):
        try:
            if debug:
                print(f"Running {run_script}")
            result = subprocess.check_output(run_script, universal_newlines=True, shell=True) # Use shell for wildcard support
            if not error_str in result:
                print(result)
                print("done")
                return True
        except subprocess.CalledProcessError:
            print("Got error!")
            continue
    return False

def ssh_execute(script, server, port, rsa_key, attempts=max_attempts):
    return make_sure_run(f"ssh -i{rsa_key} -p {port} {server} {script}", attempts=attempts)

def scp(src, dst, server, port, rsa_key):
    ssh_execute(f"mkdir -p {os.path.dirname(dst.split(':')[1])}", server, port, rsa_key)
    return make_sure_run(f"scp -i{rsa_key} -P {port} -r {src} {dst}")

def gsutil_down_folder(gsutil_src, dst, server, port, rsa_key):
    if ':' in dst:
        dst = dst.split(':')[1]
    ssh_execute(f"mkdir -p {dst}", server, port, rsa_key)
    ssh_execute(f"/home/tbvj5914/google-cloud-sdk/bin/gsutil -m cp -r {gsutil_src}/\* {dst}/", server, port, rsa_key)

def gsutil_cp(src, gsutil_tmp, dst, server, port, rsa_key):
    if ':' in dst:
        dst = dst.split(':')[1]
    if os.path.isdir(src):
        ssh_execute(f"mkdir -p {dst}", server, port, rsa_key)
        os.system(f"gsutil -m cp -r {src}/\* {gsutil_tmp}/")
        if not ssh_execute(f"/home/tbvj5914/google-cloud-sdk/bin/gsutil -m cp -r {gsutil_tmp}/\* {dst}/", server, port, rsa_key, attempts=2):
            scp(f"{src}/*", f"{server}:{dst}/", server, port, rsa_key)
    else:
        basepath = os.path.dirname(dst)
        ssh_execute(f"mkdir -p {basepath}", server, port, rsa_key)
        os.system(f"gsutil -m cp -r {src} {gsutil_tmp}")
        ssh_execute(f"/home/tbvj5914/google-cloud-sdk/bin/gsutil -m cp -r {gsutil_tmp} {basepath}/", server, port, rsa_key)
        ssh_execute(f"mv {basepath}/{os.path.split(gsutil_tmp)[1]} {dst}", server, port, rsa_key)

def scp_pull(server_src, local_dst, server, port, rsa_key):
    return make_sure_run(f"scp -i{rsa_key} -P {port} -r {server}:{server_src} {local_dst}")


def scp_push(local_src, server_dst, server, port, rsa_key):
    return make_sure_run(f"scp -i{rsa_key} -P {port} -r {local_src} {server}:{server_dst}")



class SSHClient:
    def __init__(self, server, port, rsa_key):
        rsa_key = os.path.expanduser(rsa_key)
        self.ssh_args = dict(server=server,
                        port=port,
                        rsa_key=rsa_key)
        self.verbose = True

    def execute(self, script):
        if self.verbose:
            print(f"EXECUTING {script}")
        ssh_execute(script, **self.ssh_args)

    def push(self, local_src, server_dst):
        print(f"COPYING {local_src} to {self.ssh_args['server']}:{server_dst}")
        scp_push(local_src, server_dst, **self.ssh_args)

    def pull(self, server_src, local_dst):
        print(f"COPYING {self.ssh_args['server']}:{server_src} to {local_dst}")
        scp_pull(server_src, local_dst, **self.ssh_args)


def maybe_make_rsa_key(rsa_key):
    if not os.path.exists(rsa_key):
        os.system('ssh-keygen -t rsa')


def push_to_authorized_keys(rsa_key, port, server):
    os.system(
        f'cat {rsa_key + ".pub"} | ssh -p{port} {server} "cat >> ~/.ssh/authorized_keys"'
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('server', default='root@dilab2.snu.ac.kr', type=str, help="server, like root@a.b.kr")
    parser.add_argument('-p', '--port', type=int, default=11001)
    parser.add_argument('--rsa_key', type=str, default='~/.ssh/dilab2')
    args = parser.parse_args()

    args.rsa_key = os.path.expanduser(args.rsa_key)
    maybe_make_rsa_key(args.rsa_key)
    push_to_authorized_keys(args.rsa_key, args.port, args.server)
