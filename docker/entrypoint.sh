#!/bin/sh
set -eu

# Preload known_hosts entries for smoother SSH-based MPI launches.
for host in n1 n2 n3 n4; do
    ssh-keyscan -H "$host" >> /root/.ssh/known_hosts 2>/dev/null || true
done

# Start SSH daemon required by OpenMPI remote process launch.
/usr/sbin/sshd

# Keep container alive for interactive development and repeated test runs.
exec tail -f /dev/null
