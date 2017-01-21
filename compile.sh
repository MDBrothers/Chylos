g++ -std=c++11 -fstrict-aliasing -O3 -ffast-math -march=native -mavx -fopenmp -I/home/mbrothers/Projects/Trilinos/include -L/home/mbrothers/Projects/Trilinos/lib -I./ -lkokkoscore -lteuchoscore chylos_microbenchmark.cpp

#g++ -std=c++11 -fopenmp -I/home/mbrothers/Projects/Trilinos/include -L/home/mbrothers/Projects/Trilinos/lib -I./ -lteuchoscore -lkokkoscore chylos_microbenchmark.cpp
