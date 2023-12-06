rm -rf ./test_cases;
g++ a.cpp -o gen.out;
./gen.out 10000 100 $1 $2;
