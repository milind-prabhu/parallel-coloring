#include <bits/stdc++.h>
using namespace std;

string num_to_string(int i)
{
    string ret = "";
    while(i > 0)
    {
        ret += (i%2 == 0)?"0":"1";
        i /= 2;
    }
    return ret;
}

string pad_color(string color, int max_len)
{
    while(max_len > color.size())
    {
        color += "0";
    }
    return color;
}

string new_color(string color, vector<string> neighbor, int max_len)
{
    string new_color = "";
    color = pad_color(color, max_len);
    int new_length = ceil(log2(max_len));
    for(int i = 0; i < neighbor.size(); i++)
    {
        int diff_index = -1;
        //find index at which colors differ
        for(int j = 0; j < neighbor[i].size(); j++)
        {
            if(neighbor[i][j] != color[j])
            {
                diff_index = j;
                break;
            }
        }
        cout << diff_index << endl;
        assert(diff_index != -1);
        new_color += pad_color(num_to_string(diff_index), new_length);
        new_color += color[diff_index];
    }
    return new_color;
}
int main()
{

    vector <string> neighbor = {"010", "100"};
    string color = "11";
    int max_len = 3;
    cout << new_color(color, neighbor, max_len) << endl;
    return 0;
}