FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install OpenMPI and SSH so mpirun can launch ranks across containers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server \
    openmpi-bin \
    libopenmpi-dev \
    build-essential \
    iputils-ping \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python stack for Milestone 0 and early image-processing experiments.
RUN pip install --no-cache-dir \
    mpi4py==4.0.1 \
    numpy==1.26.4 \
    pillow==10.4.0

# Prepare SSH server and key-based auth for intra-cluster launches.
RUN mkdir -p /var/run/sshd /root/.ssh /workspace \
    && chmod 700 /root/.ssh \
    && ssh-keygen -t ed25519 -N "" -f /root/.ssh/id_ed25519 \
    && cat /root/.ssh/id_ed25519.pub >> /root/.ssh/authorized_keys \
    && chmod 600 /root/.ssh/authorized_keys

# Relax host-key checks only inside this local lab environment.
RUN printf "Host *\n    StrictHostKeyChecking no\n    UserKnownHostsFile /dev/null\n    LogLevel ERROR\n" > /etc/ssh/ssh_config

# Enable root SSH login for local MPI orchestration inside containers.
RUN sed -i 's/^#\?PermitRootLogin .*/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/^#\?PubkeyAuthentication .*/PubkeyAuthentication yes/' /etc/ssh/sshd_config \
    && sed -i 's@^#\?AuthorizedKeysFile .*@AuthorizedKeysFile .ssh/authorized_keys@' /etc/ssh/sshd_config

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
