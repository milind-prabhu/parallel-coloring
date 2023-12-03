#include<bits/stdc++.h>
using namespace std;

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

int main()
{
    srand(1);
    int n, m;
    cin >> n >> m;
    int delta;
    cin >> delta;
    assert(delta > 0);
    assert(2*m <= n* delta);
    const int list_size = 2 * ceil(log2(n));
    vector<int> adj[n+1];
    for(int i = 0; i < m; i++)
    {
        int u,v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<int> colors[n+1];
    for(int i = 1; i <= n; i++)
    {
        for(int j = 0; j < list_size; j++)
        {
            colors[i].push_back(rand()%(2*delta)+1);
            cout << colors[i][j] << endl;
        }   
        cout << endl;
    }
    vector<int> coloring = list_coloring(n, adj, colors);
    for(int i = 1; i <= n; i++)
    {
        cout << coloring[i] << endl;
    }

    return 0;
}