#include <bits/stdc++.h>
#include "mpi.h"

using namespace std;

void send_palette(vector<int> palette[], int list_size, int to, int n1, int n2)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cout << "Process " << rank << " sending to " << to << endl;
    
    vector<int> arr;
    for (int i = n1; i <= n2; i++)
    {
        for (int j = 0; j < list_size; j++)
        {
            arr.push_back(palette[i][j]);
        }
    }
    MPI_Send(&arr[0], arr.size(), MPI_INT, to, 0, MPI_COMM_WORLD);
}

void receive_palette(vector<int> palette[], int list_size, int from, int n, int size)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int q1 = from * (n / size) + 1;
    int q2 = (from + 1) * (n / size);

    vector<int> arr(list_size * (q2 - q1 + 1));
    MPI_Recv(&arr[0], arr.size(), MPI_INT, from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    cout << "Process " << rank << " receiving " << from << endl;

    for (int i = q1; i <= q2; i++)
    {
        for (int j = 0; j < list_size; j++)
        {
            palette[i].resize(list_size);
            palette[i][j] = arr[(i - q1) * list_size + j];
        }
    }

}

vector<int> list_coloring(int n, vector<int> adj[], vector<int> list_of_colors[])
{
    vector<int> coloring(n+1,-1);
    bool fail = false;
    for(int i = 1; i <= n; i++)
    {
        for(int j = 0; j < list_of_colors[i].size(); j++)
        {
            bool used = false;
            for(auto u: adj[i])
            {
                if(coloring[u] == list_of_colors[i][j])
                {
                    used = true;
                    break;
                }
            }
            if(used == false)
            {
                coloring[i] = list_of_colors[i][j];
            }
        }
        if(coloring[i] == -1)
        {
            fail = true;
            break;
        }
    }
    if(fail)
    {
        cout << "Failed coloring " << endl;
    }
    return coloring;
}

int main(int argc, char *argv[])
{

    //Start measuring execution time
    double start_time, end_time;
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int n, n1, n2, m, delta;
    int num_colors = 2 * delta;

    if(rank == 0)
        start_time = MPI_Wtime();


    //random number generator
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    


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

    double file_io_end_time;
    if(rank == 0)
        file_io_end_time = MPI_Wtime();

    // Sample color palette for each vertex
    const int list_size = min((int)(4 * ceil(log2(n))), 2 * delta);
    vector<int> palette[n];
    for (int i = n1; i <= n2; i++)
    {
        for (int j = 0; j < list_size; j++)
        {
            palette[i].push_back(rng() % (2 * delta) + 1);
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
                    send_palette(palette, list_size, p, n1, n2);
                }
                else
                {
                    send_palette(palette, list_size, p, n1, n2);
                    receive_palette(palette, list_size, p, n, size);
                }
                break;
            }
            // synchronize all processes
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }


    // We say that an edge (u,v) is a conflict edge if their palettes have a common color
    // Each processor finds the conflict edges in its assigned graph
    vector <int> conflict_edges;
    for(int i = n1; i <= n2; i++)
    {
        for(int j = 0; j < adj[i].size(); j++)
        {
            int u = i, v = adj[i][j];
            if(u > v) continue;
            for(int k = 0; k < list_size; k++)
            {
                for(int l = 0; l < list_size; l++)
                {
                    if(palette[u][k] == palette[v][l])
                    {
                        // (u,v) is a conflict edge
                        conflict_edges.push_back(u);
                        conflict_edges.push_back(v);
                        goto end;
                    }
                }
            }
            end:;
        }
    }
    if(rank != 0)
    {
            MPI_Send(&conflict_edges[0], conflict_edges.size(), MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else
    {
        //Process 0 is the coordinator. It receives all the conflict graphs and palettes. It then simply runs the greedy list colouring algorithm.
        vector <int> adj_conflict[n+1];

        for(int i = 1; i < size; i++)
        {
            MPI_Status status;
            MPI_Probe(i, 0, MPI_COMM_WORLD, &status);
            int arr_size;
            MPI_Get_count(&status, MPI_INT, &arr_size);
            vector <int> temp(arr_size);
            MPI_Recv(&temp[0], arr_size, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for(int j = 0; j < temp.size()/2; j++)
            {
                adj_conflict[temp[2*j]].push_back(temp[2*j+1]);
                adj_conflict[temp[2*j+1]].push_back(temp[2*j]);
            }
        }
        for(int i = 0; i < conflict_edges.size()/2; i++)
        {
            adj_conflict[conflict_edges[2*i]].push_back(conflict_edges[2*i+1]);
            adj_conflict[conflict_edges[2*i+1]].push_back(conflict_edges[2*i]);
        }
        vector <int> final_coloring = list_coloring(n, adj_conflict, palette);

    }
    if(rank == 0)
    {
        end_time = MPI_Wtime();
        cout <<"---------------------------------------------------------------------------------------" << endl;
        cout << "File IO time: " << file_io_end_time - start_time << endl;
        cout << "Execution time: " << end_time - file_io_end_time << endl;
        cout << "Total time: " << end_time - start_time << endl;
    }
    MPI_Finalize();

    return 0;
}