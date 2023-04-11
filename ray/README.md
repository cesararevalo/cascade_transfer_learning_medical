# Ray cluster

1. Install ray locally:
```
% python3 -m virtualenv --python=python3.10 venv # setup a venv with python 3.10
% source venv/bin/activate
% pip install -U "ray[default]" boto3
```

2. Start the ray cluster:
```
% cd ray/
% ray up -y config.yaml
```

3. Submit the test script (from the ray getting started guide):
```
% ray submit config.yaml script.py
```

3. Shutdown the cluster:
```
% ray down -y config.yaml
```

# Building Docker Image

* CPU Image:
```
% docker build --file Dockerfile_cpu ../code/
% docker tag 9a852d980f91 cesararevalo/mcs:ray_cpu
% docker push cesararevalo/mcs:ray_cpu
```

* GPU Image:
```
% docker build --file Dockerfile_gpu ../code/
% docker tag 4cff86c3ca91 cesararevalo/mcs:ray_gpu
% docker push cesararevalo/mcs:ray_gpu
```

# References:
* https://docs.ray.io/en/latest/cluster/vms/getting-started.html#vm-cluster-quick-start
* https://github.com/ray-project/ray/blob/master/python/ray/autoscaler/aws/example-full.yaml
