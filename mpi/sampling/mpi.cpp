#include <bits/stdc++.h>
#include "mpi.h"

using namespace std;

void send_palette(vector<int> palette[], int list_size, int to, int n1, int n2)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cout << "Process " << rank << " sending";
    
    vector<int> arr;
    for (int i = n1; i <= n2; i++)
    {
        for (int j = 0; j < list_size; j++)
        {
            arr.push_back(palette[i][j]);
        }
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));


    //std::this_thread::sleep_for(std::chrono::seconds(1));
    MPI_Send(&arr[0], arr.size(), MPI_INT, to, 0, MPI_COMM_WORLD);
}

void receive_palette(vector<int> palette[], int list_size, int from, int n, int size)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int q1 = from * (n / size) + 1;
    int q2 = (from + 1) * (n / size);

    vector<int> arr(list_size * (q2 - q1 + 1));
    cout << "Waiting..." << endl;
    MPI_Recv(&arr[0], arr.size(), MPI_INT, from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    cout << "Process " << rank << " recv" << from << endl;


    for (int i = q1; i <= q2; i++)
    {
        for (int j = 0; j < list_size; j++)
        {
            palette[i][j] = arr[(i - q1) * list_size + j];
            cout << palette[i][j] << endl;
        }
    }
}

int main(int argc, char *argv[])
{
    srand(12);
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int n, n1, n2, m, delta;
    int num_colors = 2 * delta;

    // Each process reads data from its assigned file
    string file_name = "graph" + to_string(rank) + ".txt";
    ifstream file(file_name);

    string data;
    if (file.is_open())
    {
        getline(file, data);
        stringstream vals(data);
        vals >> n >> n1 >> n2 >> m >> delta;
    }
    else
    {
        cerr << "Unable to open file: " << file_name << endl;
    }
    vector<int> adj[n + 1];

    while (getline(file, data))
    {
        stringstream buffer(data);
        int u, v;
        buffer >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    file.close();

    // Sample color palette for each vertex
    const int list_size = min((int)(4 * ceil(log2(n))), 2 * delta);
    vector<int> palette[n];
    for (int i = n1; i <= n2; i++)
    {
        for (int j = 0; j < list_size; j++)
        {
            palette[i].push_back(rand() % (2 * delta) + 1);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Send palette to neighbors - this requires synchronization
    // The idea is to edge color the processor graph which is a clique.
    // Then processors the round in which processors (i,j) communicate is determined by the color of the edge (i,j)

    // We asssume that the processor graph is a clique with "size" vertices- therefore it can be edge colored with "size" colors
    // Moreover, color(i,j) = (i+j)%size is a valid coloring
    for (int round = 0; round < size; round++)
    {
        for (int p = 0; p < size; p++)
        {
            if (p == rank)
                continue;
            if ((p + rank) % size == round)
            {
                if (rank < p)
                {
                    receive_palette(palette, list_size, p, n, size);
                    cout <<"step 1 done" << endl;
                        std::this_thread::sleep_for(std::chrono::seconds(4));

                    send_palette(palette, list_size, p, n1, n2);
                }
                else
                {
                    send_palette(palette, list_size, p, n1, n2);
                    cout <<"step 1 done" << endl;
                    std::this_thread::sleep_for(std::chrono::seconds(4));

                    receive_palette(palette, list_size, p, n, size);
                }
                break;
            }
            // synchronize all processes
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }


    // Creating the conflict graph

    // MPI_finalize();

    return 0;
}