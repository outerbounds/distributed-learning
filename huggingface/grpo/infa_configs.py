AWS_H100_CONFIG = dict(
    cpu=150,
    memory=1500 * 1000,
    gpu=8,
    shared_memory=300 * 1000,
    image="registry.hub.docker.com/valayob/pytorch-efa-base:0.0.2",
    disk=100 * 1000,
    extended_resources={"vpc.amazonaws.com/efa": 32},
)

AWS_G4_CONFIG = dict(
    cpu=40,
    memory=200 * 1000,
    gpu=4,
    shared_memory=40 * 1000,
    disk=80 * 1000,
    compute_pool="obp-g24x-nonhpc",
    image="registry.hub.docker.com/valayob/pytorch-efa-base:0.0.2",
    # extended_resources={"vpc.amazonaws.com/efa": 1},
)


H100_K8S_CONFIG = dict(
    cpu=100,
    memory=900 * 1000,
    gpu=8,
    shared_memory=200 * 1000,
    image="registry.hub.docker.com/valayob/nebius-nccl-pytorch:0.0.2",
    # This thing needs a security context of `V1Container` with privilage=true to use Infiniband.
    disk=1000 * 1000,
    use_tmpfs=True,
)
