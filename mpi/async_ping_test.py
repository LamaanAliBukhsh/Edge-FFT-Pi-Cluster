from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Ring topology: each rank sends to next rank and receives from previous rank.
send_to = (rank + 1) % size
recv_from = (rank - 1) % size

send_buf = np.array([rank], dtype=np.int32)
recv_buf = np.empty(1, dtype=np.int32)

req_send = comm.Isend(send_buf, dest=send_to, tag=7)
req_recv = comm.Irecv(recv_buf, source=recv_from, tag=7)

MPI.Request.Waitall([req_send, req_recv])
print(f"[async] rank={rank} received={recv_buf[0]} from={recv_from} sent_to={send_to}")
