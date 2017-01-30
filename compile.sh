g++ -std=c++11 -fassociative-math -Ofast -march=native -mavx -fopenmp -I/home/mbrothers/Projects/Trilinos/include -L/home/mbrothers/Projects/Trilinos/lib -I./ -lkokkoscore chylos_microbenchmark.cpp

#g++ -std=c++11 -fopenmp -I/home/mbrothers/Projects/Trilinos/include -L/home/mbrothers/Projects/Trilinos/lib -I./ -lkokkoscore chylos_microbenchmark.cpp
