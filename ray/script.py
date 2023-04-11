# Getting Started script from https://docs.ray.io/en/latest/cluster/vms/getting-started.html#vm-cluster-quick-start
from collections import Counter
import socket
import time

import ray

ray.init(address='auto')

@ray.remote(num_gpus=1)
def f():
    time.sleep(0.001)
    # Return IP address.
    return socket.gethostbyname(socket.gethostname())

object_ids = [f.remote() for _ in range(10000)]
ip_addresses = ray.get(object_ids)

print('''This cluster consists of
    {} nodes in total
    {} CPU resources in total
'''.format(len(ray.nodes()), ray.cluster_resources()['CPU']))

print('Tasks executed')
for ip_address, num_tasks in Counter(ip_addresses).items():
    print('    {} tasks on {}'.format(num_tasks, ip_address))