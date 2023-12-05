#include <bits/stdc++.h>
using namespace std;
int main(int argc, char** argv)
{
    assert(argc == 4);
    int n = atoi(argv[1]);
    int d = atoi(argv[2]);
    int seed = atoi(argv[3]);
    srand(seed);
    vector <int> u,v;
    vector <int> degree(n+1,0);

    for(int i = 1; i <= n; i++)
    {
        for(int j = i+1; j <= n; j++)
        {
            if((rand() % n) < d)
            {
                u.push_back(i);
                v.push_back(j);
                degree[i]++;
                degree[j]++;
            }
        }
    }
    cout << n << " " << u.size() << endl;

    for(int i = 0; i < u.size(); i++)
    {
        cout << u[i] << " " << v[i] << endl;
    }

    int max_degree = -1;
    for(int i = 1; i <= n;i++)
    {
        max_degree = max(max_degree, degree[i]);
    }
    //cout << max_degree << endl;



    return 0;
}