#include <bits/stdc++.h>
using namespace std;
int main()
{
    int size = 5;
    for(int i = 0; i < size; i++)
    {
        for(int j = 0; j < size; j++)
        {
            if(i != j)
                cout << (i+j) % (size) << " " ;
            else
                cout << -1 << " ";
        }
        cout << endl;
    } 
}