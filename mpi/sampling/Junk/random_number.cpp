#include <bits/stdc++.h>
using namespace std;
int main()
{
     mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
     vector <int> cnt(10,0);
    for(int i = 0; i < 10000; i++)
    {
        cnt[rng() % 10]++;
    }
    for(int i = 0; i < 10; i++) cout << cnt[i] << " ";
    return 0;
}