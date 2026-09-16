// SIT315 Seminar 7 - Activity 1 (MPI communication)
// Extends the Seminar 6 hello_mpi.cpp in two steps:
//   Part 1: master sends "Hello World!" to each worker with MPI_Send,
//           each worker receives it with MPI_Recv and prints it.
//   Part 2: the same message goes out with a single MPI_Bcast.
//
// Build:  mpic++ hello_mpi_comm.cpp -o hello_mpi_comm
// Run:    mpirun -np 4 --hostfile hostfile ./hello_mpi_comm

#include <mpi.h>
#include <stdio.h>
#include <string.h>

#define MSG_LEN 32

int main(int argc, char** argv) {
    int numtasks, rank, name_len, tag = 1;
    char name[MPI_MAX_PROCESSOR_NAME];
    char msg[MSG_LEN];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(name, &name_len);

    // ---------- Part 1: point-to-point with MPI_Send / MPI_Recv ----------
    if (rank == 0) {
        strcpy(msg, "Hello World!");
        // Master sends one message per worker. Each MPI_Send is matched by
        // exactly one MPI_Recv on the destination rank with the same tag.
        for (int dest = 1; dest < numtasks; dest++) {
            MPI_Send(msg, MSG_LEN, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
            printf("[Send/Recv] master (rank 0, %s) sent to rank %d\n", name, dest);
        }
    } else {
        // Worker blocks here until the master's message lands in msg.
        MPI_Recv(msg, MSG_LEN, MPI_CHAR, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[Send/Recv] worker (rank %d of %d, %s) received: %s\n",
               rank, numtasks, name, msg);
    }

    // Keep the two parts' output from interleaving on screen.
    MPI_Barrier(MPI_COMM_WORLD);

    // ---------- Part 2: collective with MPI_Bcast ----------
    // Wipe the buffer on the workers so the printed text can only have come
    // from the broadcast, not left over from Part 1.
    memset(msg, 0, MSG_LEN);
    if (rank == 0) {
        strcpy(msg, "Hello World!");
    }

    // Every rank calls MPI_Bcast. Rank 0 (the root) supplies the data, the
    // other ranks receive into their own msg buffer. One call replaces the
    // send loop and the library handles the fan-out.
    MPI_Bcast(msg, MSG_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("[Bcast] master (rank 0, %s) broadcast: %s\n", name, msg);
    } else {
        printf("[Bcast] worker (rank %d of %d, %s) received: %s\n",
               rank, numtasks, name, msg);
    }

    MPI_Finalize();
    return 0;
}