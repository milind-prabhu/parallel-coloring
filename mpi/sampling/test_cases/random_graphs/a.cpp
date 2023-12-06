#include <bits/stdc++.h>
using namespace std;
int main(int argc, char** argv)
{
    assert(argc == 5);
    int n = atoi(argv[1]);
    int d = atoi(argv[2]);
    int size = atoi(argv[3]);
    int seed = atoi(argv[4]);
    assert(n%size == 0);
    srand(seed);
    vector <int> adj[n+1];
    vector <int> degree(n+1,0);
    int m = 0;

    for(int i = 1; i <= n; i++)
    {
        for(int j = i+1; j <= n; j++)
        {
            if((rand() % n) < d)
            {
                adj[i].push_back(j);
                adj[j].push_back(i);
                degree[i]++;
                degree[j]++;
                m++;
            }
        }
    }

    int max_degree = -1;
    for(int i = 1; i <= n;i++)
    {
        max_degree = max(max_degree, degree[i]);
    }

    //cout << max_degree << endl;


    //code to create a directory ./test_cases in the current directory
    //if the directory already exists, it will do nothing
    system("rm -rf test_cases");
    system("mkdir -p test_cases");
    for(int i = 0; i < size; i++)
    {
        ofstream fout("test_cases/graph" + to_string(i) + ".txt");
        int n1 = i * (n / size) + 1;
        int n2 = (i + 1) * (n / size);
        fout << n << " " << n1 << " " << n2 << " " << m << " " << max_degree << endl;
        for(int j = n1; j <= n2; j++)
        {
            for(auto u: adj[j])
            {
                fout << j << " " << u << endl;
            }
        }
        fout.close();
    }
    ofstream fullgraph("test_cases/full_graph.txt");
    for(int i = 1; i <= n; i++)
    {
        for(auto u: adj[i])
        {
            fullgraph << i << " " << u << endl;
        }
    }
    fullgraph.close();
    return 0;
}