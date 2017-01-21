#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <Kokkos_Core.hpp>
#include <cmath>
#include <iostream>
#include "Teuchos_RCP.hpp"
#include <vector>

#define SCALAR_DATA_TYPE double


template <typename E>
class ArrayExpression {
  public:
    SCALAR_DATA_TYPE operator[](const size_t i) const { return static_cast<E const&>(*this)[i];}
    // size_t size()          const { return static_cast<E const&>(*this).size(); }

    // The following overload conversions to E, the template argument type;
    // e.g., for ArrayExpression<ArraySum>, this is a conversion to ArraySum.
          E& operator()()       { return static_cast<      E&>(*this); }
    const E& operator()() const { return static_cast<const E&>(*this); }
};


template <size_t N>
class Array : public ArrayExpression<Array<N> > {
    typedef Kokkos::View<double[N]> view_type;
    std::vector<SCALAR_DATA_TYPE> elems;

  public:
    SCALAR_DATA_TYPE operator[](const size_t i) const { return elems[i]; }
    SCALAR_DATA_TYPE &operator[](const size_t i)      { return elems[i]; }

    //  constructors
    Array(const SCALAR_DATA_TYPE initial_value) {
      elems.resize(N);
      Kokkos::parallel_for( N, [&] ( size_t i ) {
            elems[i] = initial_value;
        });
    }
    Array() {elems.resize(N);}

    // Assignment with evaluation
    template <typename E>
    Array & operator=(ArrayExpression<E> const& arr) {
      Kokkos::parallel_for( N, [&] ( size_t i ) {
        elems[i] = arr[i];
      });
      return *this;
    }

};

template<typename E1, typename E2>
class ArraySum : public ArrayExpression< ArraySum<E1,E2> > {
    E1 const& _u;
    E2 const& _v;

  public:
     ArraySum(E1 const& u, E2 const& v) : _u(u), _v(v) {
   }
    SCALAR_DATA_TYPE operator[](size_t i) const {return _u[i] + _v[i]; }
};

template <typename E1, typename E2>
inline ArraySum<E1, E2> const operator+(E1 const& u, E2 const& v) {
   return ArraySum<E1, E2>(u, v);
}

int main(int argc, char* argv[])
{

  const int N = 10000000;
  const int nrepeat = 100;

  Kokkos::initialize(argc,argv);

  Array<N> a(1000.0);
  Array<N> b(111.1);

  // Timer products
  struct timeval begin,end;

  std::cout << a[3] << std::endl;
  gettimeofday(&begin,NULL);

  for(int i = 0; i < nrepeat; i++)
    a = a + b + a + b;

    gettimeofday(&end,NULL);
  std::cout << a[3] << std::endl;

  // Calculate time
  double time = 1.0*(end.tv_sec-begin.tv_sec) + 1.0e-6*(end.tv_usec-begin.tv_usec);

  // Calculate bandwidth.

  // Print results (problem size, time and bandwidth in GB/s)
  printf("time( %g s )\n", time );

  Kokkos::finalize();

  return 0;
}
