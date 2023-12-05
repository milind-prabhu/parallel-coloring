#include <bits/stdc++.h>
#include "mpi.h"
#include "f.h"
using namespace std;


int n, p, sp, m;

bool myblock(int i, int j, int sp, int row, int col)
{
    int r = i / sp;
    int c = j / sp;
    return (r == row && c == col);
}

int which_proc(int i, int j, int m, int sp)
{
    int r = i / m;
    int c = j / m;
    return r * sp + c;
}

int main(int argc, char *argv[])
{
    assert(argc == 2);
    n = atoi(argv[1]);

    int rank;
    double startwtime = 0.0, endwtime;
    int namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &p);
    sp = sqrt(p);
    m = n / sp;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(processor_name, &namelen);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        startwtime = MPI_Wtime();
    // Start

    int row = rank / sp;
    int col = rank % sp;

    vector<vector<long long>> block(m + 2, vector<long long>(m + 2, 0));
    vector<vector<long long>> block2(m + 2, vector<long long>(m + 2, 0));

    // Initialization
    for (int i = row * m; i < (row + 1) * m; i++)
    {
        for (int j = col * m; j < (col + 1) * m; j++)
        {
            block[i - row * m + 1][j - col * m + 1] = i  + j*n;
            block2[i - row * m + 1][j - col * m + 1] = i  + j*n;
        }
    }

    int iteration = 0;
    while (iteration < 10)
    {
        // communicate values

        // right to left - even send
        if (col % 2 == 0)
        {
            if (col != 0)
            {
                for (int i = 1; i <= m; i++)
                {
                    MPI_Send(&block[i][1], 1, MPI_LONG_LONG, rank - 1, 0, MPI_COMM_WORLD);
                }
            }
        }
        // right to left - odd receive
        if (col % 2 == 1)
        {
            if (col != sp - 1)
            {
                for (int i = 1; i <= m; i++)
                {
                    MPI_Recv(&block[i][m + 1], 1, MPI_LONG_LONG, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }

        // right to left - odd send
        if (col % 2 == 1)
        {
            if (col != 0)
            {
                for (int i = 1; i <= m; i++)
                {
                    MPI_Send(&block[i][1], 1, MPI_LONG_LONG, rank - 1, 0, MPI_COMM_WORLD);
                }
            }
        }
        // right to left - even receive
        if (col % 2 == 0)
        {
            if (col != sp - 1)
            {
                for (int i = 1; i <= m; i++)
                {
                    MPI_Recv(&block[i][m + 1], 1, MPI_LONG_LONG, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }

        // bottom to top- even send
        if (row % 2 == 0)
        {
            if (row != 0)
            {
                for (int i = 1; i <= m; i++)
                {
                    MPI_Send(&block[1][i], 1, MPI_LONG_LONG, rank - sp, 0, MPI_COMM_WORLD);
                }
            }
        }
        // bottom to top - odd receive
        if (row % 2 == 1)
        {
            if (row != sp - 1)
            {
                for (int i = 1; i <= m; i++)
                {
                    MPI_Recv(&block[m + 1][i], 1, MPI_LONG_LONG, rank + sp, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }

        // bottom to top- odd send
        if (row % 2 == 1)
        {
            if (row != 0)
            {
                for (int i = 1; i <= m; i++)
                {
                    MPI_Send(&block[1][i], 1, MPI_LONG_LONG, rank - sp, 0, MPI_COMM_WORLD);
                }
            }
        }
        // bottom to top - even receive
        if (row % 2 == 0)
        {
            if (row != sp - 1)
            {
                for (int i = 1; i <= m; i++)
                {
                    MPI_Recv(&block[m + 1][i], 1, MPI_LONG_LONG, rank + sp, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }

        // diagonal - even send
        if (row % 2 == 0 && (row != 0 && col != 0))
        {
            MPI_Send(&block[1][1], 1, MPI_LONG_LONG, rank - sp - 1, 0, MPI_COMM_WORLD);
        }
        // diagonal - odd receive
        if (row % 2 == 1 && row != sp - 1 && col != sp - 1)
        {
            MPI_Recv(&block[m + 1][m + 1], 1, MPI_LONG_LONG, rank + sp + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // diagonal - odd send
        if (row % 2 == 1 && (row != 0 && col != 0))
        {
            MPI_Send(&block[1][1], 1, MPI_LONG_LONG, rank - sp - 1, 0, MPI_COMM_WORLD);
        }

        // diagonal - even receive
        if (row % 2 == 0 && row != sp - 1 && col != sp - 1)
        {
            MPI_Recv(&block[m + 1][m + 1], 1, MPI_LONG_LONG, rank + sp + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
 
        // compute values
        // if(rank == 2)
        // {
        //     for(int i = 1; i <= m; i++)
        //     {
        //         for(int j = 1; j <= m; j++)
        //         {
        //             cout << block2[i][j] << " ";
        //         }
        //         cout << "\n";
        //     }
        // }
        for (int i = 1; i <= m; i++)
        {
            for (int j = 1; j <= m; j++)
            {
                long long a = block[i][j];
                long long b = block[i + 1][j];
                long long c = block[i][j + 1];
                long long d = block[i + 1][j + 1];

                int true_row = row * m + i - 1;
                int true_col = col * m + j - 1;

                if (true_row == 0 || true_row == n - 1 || true_col == 0 || true_col == n - 1)
                {
                    continue;
                }
                block2[i][j] = f(a, b, c, d);
            }
        }

        for (int i = 1; i <= m ; i++)
        {
            for (int j = 1; j <= m ; j++)
            {
                block[i][j] = block2[i][j];
                
            }
        }
        // if(rank == 2)
        // {
        //     for(int i = 1; i <= m; i++)
        //     {
        //         for(int j = 1; j <= m; j++)
        //         {
        //             cout << block[i][j] << " ";
        //         }
        //         cout << "\n";
        //     }
        // }

        iteration++;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Check sumj
    long long sum = 0;
    for (int i = 1; i <= m; i++)
    {
        for (int j = 1; j <= m; j++)
        {
            sum += block[i][j];
            //cout << block[i][j] << " ";
        }
        // cout << "\n";
    }
    if (rank != 0)
    {
        MPI_Send(&sum, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }
    else
    {
        long long total_sum = sum;
        for (int i = 1; i < p; i++)
        {

            MPI_Recv(&sum, 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (i == 1)
            {
                //cout << sum << "\n";
            }
            total_sum += sum;
        }
        cout << "Total Sum = " << total_sum << "\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);


    // Check A(n/3, 2n/3)
    int spl_rank = which_proc(n / 3, 2 * n / 3, m, sp);
    if (rank == spl_rank)
    {
        int true_row = n / 3;
        int true_col = (2 * n) / 3;
        int row = true_row % m;
        int col = true_col % m;
        cout << "A(n/3, 2n/3) = " << block[row + 1][col + 1] << "\n";

        for (int i = 0; i < p; i++)
        {
            if (i != rank)
            {
                MPI_Send(&block[row + 1][col + 1], 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD);
            }
        }
        int count = 0;
        for (int x = 1; x <= m; x++)
        {
            for (int y = 1; y <= m; y++)
            {
                if (block[x][y] == block[row + 1][col + 1])
                {
                    count++;
                }
            }
        }
        for (int i = 0; i < p; i++)
        {
            int temp;
            if (i != rank)
            {
                MPI_Recv(&temp, 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                count += temp;
            }
        }
        cout << "Number of entries equal to A(n/3, 2n/3) = " << count << "\n";
    }
    else
    {
        // Count number of entries with A(n/3, 2n/3)
        long long temp_val;
        MPI_Recv(&temp_val, 1, MPI_LONG_LONG, spl_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int local_count = 0;
        for (int i = 1; i <= m; i++)
        {
            for (int j = 1; j <= m; j++)
            {
                if (block[i][j] == temp_val)
                {
                    local_count++;
                }
            }
        }
        MPI_Send(&local_count, 1, MPI_INT, spl_rank, 0, MPI_COMM_WORLD);
    }

    // End
    MPI_Barrier(MPI_COMM_WORLD);
    // Measure time
    if (rank == 0)
    {
        endwtime = MPI_Wtime();
        printf("Time Spent = %f\n", endwtime - startwtime);
        fflush(stdout);
    }
    // printf("outside loop....\n");
    fflush(stdout);
    return 0;
}