#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <Kokkos_Core.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include <memory>
#include <typeinfo>
#include <initializer_list>

#define SCALAR_DATA_TYPE double

typedef Kokkos::OpenMP ExecSpace;
typedef Kokkos::OpenMP MemSpace;
typedef Kokkos::RangePolicy<ExecSpace> range_policy;
typedef Kokkos::View<SCALAR_DATA_TYPE*, MemSpace> view_type;
char namechar[1] = {'a'};

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

template <typename E>
struct assignment_kernel{
  view_type _local_view;
  ArrayExpression<E> const& _local_arr_ref;
  assignment_kernel(view_type my_view_, ArrayExpression<E> const& arr_): _local_view(my_view_), _local_arr_ref(arr_){}
  KOKKOS_INLINE_FUNCTION
  void operator() (const typename view_type::size_type i) const {
    _local_view(i) = _local_arr_ref[i];
  }
};

template <typename E>
struct assignment_w_add_kernel{
  view_type _local_view;
  ArrayExpression<E> const& _local_arr_ref;
  assignment_w_add_kernel(view_type my_view_, ArrayExpression<E> const& arr_): _local_view(my_view_), _local_arr_ref(arr_){}
  KOKKOS_INLINE_FUNCTION
  void operator() (const typename view_type::size_type i) const {
    _local_view(i) += _local_arr_ref[i];
  }
};

template <typename E>
struct assignment_w_subtract_kernel{
  view_type _local_view;
  ArrayExpression<E> const& _local_arr_ref;
  assignment_w_subtract_kernel(view_type my_view_, ArrayExpression<E> const& arr_): _local_view(my_view_), _local_arr_ref(arr_){}
  KOKKOS_INLINE_FUNCTION
  void operator() (const typename view_type::size_type i) const {
    _local_view(i) -= _local_arr_ref[i];
  }
};

template <typename E>
struct assignment_w_multiply_kernel{
  view_type _local_view;
  ArrayExpression<E> const& _local_arr_ref;
  assignment_w_multiply_kernel(view_type my_view_, ArrayExpression<E> const& arr_): _local_view(my_view_), _local_arr_ref(arr_){}
  KOKKOS_INLINE_FUNCTION
  void operator() (const typename view_type::size_type i) const {
     _local_view(i) *= _local_arr_ref[i];
  }
};

template <typename E>
struct assignment_w_divide_kernel{
  view_type _local_view;
  ArrayExpression<E> const& _local_arr_ref;
  assignment_w_divide_kernel(view_type my_view_, ArrayExpression<E> const& arr_): _local_view(my_view_), _local_arr_ref(arr_){}
  KOKKOS_INLINE_FUNCTION
  void operator() (const typename view_type::size_type i) const {
    _local_view(i) /= _local_arr_ref[i];
  }
};


template <typename E>
struct compound_assignment_kernel{
  view_type _local_view;
  E const& _local_arr_ref;
  compound_assignment_kernel(view_type my_view_, E const& arr_): _local_view(my_view_), _local_arr_ref(arr_){}
  KOKKOS_INLINE_FUNCTION
  void operator() (const typename view_type::size_type i) const {
    _local_view(i) = _local_arr_ref[i] + _local_arr_ref[i] +  _local_arr_ref[i] + _local_arr_ref[i];
  }
};

template <size_t N>
class Array : public ArrayExpression<Array<N> > {
  view_type my_view;

  public:
    SCALAR_DATA_TYPE operator[](const size_t i) const { return my_view(i); }
    SCALAR_DATA_TYPE &operator[](const size_t i)      { return my_view(i); }
    // Assignment with evaluation
    template <typename E>
    inline Array & operator=(ArrayExpression<E> const& arr) {
      Kokkos::parallel_for( range_policy(0,N), assignment_kernel<E>(my_view,arr) );
      return *this;
    }
    // Assignment with addition
    template <typename E>
    inline Array & operator+=(ArrayExpression<E> const& arr) {
      Kokkos::parallel_for( range_policy(0,N), assignment_w_add_kernel<E>(my_view,arr) );
      return *this;
    }
    // Assignment with subtraction
    template <typename E>
    inline Array & operator-=(ArrayExpression<E> const& arr) {
      Kokkos::parallel_for( range_policy(0,N), assignment_w_subtract_kernel<E>(my_view,arr) );
      return *this;
    }
    // Assignment with multiplication
    template <typename E>
    inline Array & operator*=(ArrayExpression<E> const& arr) {
      Kokkos::parallel_for( range_policy(0,N), assignment_w_multiply_kernel<E>(my_view,arr) );
      return *this;
    }
    // Assignment with division
    template <typename E>
    inline Array & operator/=(ArrayExpression<E> const& arr) {
      Kokkos::parallel_for( range_policy(0,N), assignment_w_divide_kernel<E>(my_view,arr) );
      return *this;
    }
    // compound_assignment
    inline Array & sum_4_then_assign(Array<N> const& arr_vec) {
      Kokkos::parallel_for( range_policy(0,N), compound_assignment_kernel<Array<N> >(my_view,arr_vec) );
      return *this;
    }
    //  constructors
    Array(const SCALAR_DATA_TYPE initial_value) {
      my_view = view_type(namechar, N);
      namechar[0]++;
      Kokkos::parallel_for( range_policy(0,N) , KOKKOS_LAMBDA ( size_t i ) {
            my_view(i) = initial_value;
        });
        std::cout << initial_value << " " << namechar[0] << std::endl;

    }
    Array() {
      my_view = view_type(namechar, N);
      std::cout << namechar[0] << std::endl;
      namechar[0]++;
    }
    Array(const Array& other) {
      my_view = view_type(namechar, N);
      namechar[0]++;
      Kokkos::parallel_for( range_policy(0,N) , KOKKOS_LAMBDA ( size_t i ) {
            my_view(i) = other[i];
        });

    }
};

template<typename E1, typename E2>
class ArrayAdd : public ArrayExpression< ArrayAdd<E1,E2> > {
    E1 const& _u;
    E2 const& _v;
  public:
     ArrayAdd(E1 const& u, E2 const& v) : _u(u), _v(v) {
   }
    SCALAR_DATA_TYPE operator[](size_t i) const {return _u[i] + _v[i]; }
};
template <typename E1, typename E2>
inline ArrayAdd<ArrayExpression<E1>, ArrayExpression<E2> > const operator+(ArrayExpression<E1> const& u, ArrayExpression<E2> const& v) {
   return ArrayAdd<ArrayExpression<E1>, ArrayExpression<E2> >(u, v);
}

template<typename E1, typename E2>
class ArraySubtract : public ArrayExpression< ArraySubtract<E1,E2> > {
    E1 const& _u;
    E2 const& _v;
  public:
     ArraySubtract(E1 const& u, E2 const& v) : _u(u), _v(v) {
   }
    SCALAR_DATA_TYPE operator[](size_t i) const {return _u[i] - _v[i]; }
};
template <typename E1, typename E2>
inline ArraySubtract<ArrayExpression<E1>, ArrayExpression<E2> > const operator-(ArrayExpression<E1> const& u, ArrayExpression<E2> const& v) {
   return ArraySubtract<ArrayExpression<E1>, ArrayExpression<E2> >(u, v);
}

template<typename E1, typename E2>
class ArrayMultiply : public ArrayExpression< ArrayMultiply<E1,E2> > {
    E1 const& _u;
    E2 const& _v;
  public:
     ArrayMultiply(E1 const& u, E2 const& v) : _u(u), _v(v) {
   }
    SCALAR_DATA_TYPE operator[](size_t i) const {return _u[i] * _v[i]; }
};
template <typename E1, typename E2>
inline ArrayMultiply<ArrayExpression<E1>, ArrayExpression<E2> > const operator*(ArrayExpression<E1> const& u, ArrayExpression<E2> const& v) {
   return ArrayMultiply<ArrayExpression<E1>, ArrayExpression<E2> >(u, v);
}

template<typename E1, typename E2>
class ArrayDivide : public ArrayExpression< ArrayDivide<E1,E2> > {
    E1 const& _u;
    E2 const& _v;
  public:
     ArrayDivide(E1 const& u, E2 const& v) : _u(u), _v(v) {
   }
    SCALAR_DATA_TYPE operator[](size_t i) const {return _u[i] / _v[i]; }
};
template <typename E1, typename E2>
inline ArrayDivide<ArrayExpression<E1>, ArrayExpression<E2> > const operator/(ArrayExpression<E1> const& u, ArrayExpression<E2> const& v) {
   return ArrayDivide<ArrayExpression<E1>, ArrayExpression<E2> >(u, v);
}

template<typename E>
class ArraySin : public ArrayExpression< ArraySin<E> > {
    E const& _u;
  public:
     ArraySin(E const& u) : _u(u) {
   }
    SCALAR_DATA_TYPE operator[](size_t i) const {return sin(_u[i]); }
};
template <typename E>
inline ArraySin<ArrayExpression<E> > const sin(ArrayExpression<E> const& u) {
   return ArraySin<ArrayExpression<E> >(u);
}

inline void basic_kernel_ab_p_ab(std::vector<SCALAR_DATA_TYPE>& my_a, const std::vector<SCALAR_DATA_TYPE>& my_b){
  const size_t a_length=my_a.size();
  for(size_t my_index=0; my_index < a_length; my_index++){
    // my_a[my_index] = sin(my_a[my_index]*(my_c[my_index] + my_b[my_index]));
    my_a[my_index] = my_b[my_index] + my_b[my_index]+ my_b[my_index]+ my_b[my_index];
  }
}

int main(int argc, char* argv[])
{

  const int N = exp2(2);
  const int nrepeat = 1;
  const int mrepeat = 1;
  const double num_ops = N*2*double(nrepeat*mrepeat);


  Kokkos::initialize(argc,argv);

  std::vector<SCALAR_DATA_TYPE> a_basic(N,0.00000001);
  std::vector<SCALAR_DATA_TYPE> b_basic(N,0.2);
  Array<N> a(0.00000001);
  Array<N> b(0.2);

  // Timer products
  struct timeval begin,end;

  for(int outer_repeat(0); outer_repeat< mrepeat; outer_repeat++){
  //Test Chylos
  gettimeofday(&begin,NULL);
  for(int i = 0; i < nrepeat; i++){
    // a = sin(a*(b+c));
    a.sum_4_then_assign(b);
  }
  gettimeofday(&end,NULL);
  // Calculate time
  double time = 1.0*(end.tv_sec-begin.tv_sec) + 1.0e-6*(end.tv_usec-begin.tv_usec);
  double flops_chylos(0.0), flops_base(0.0);
  // Calculate time and flops.
  printf("\nChylos implementation time( %g s )\n", time );
  printf("Chylos implementation flops( %g GFlops )\n\n", (flops_chylos = 1.e-9*num_ops/time, flops_chylos) );

  // Test base implementation
  gettimeofday(&begin,NULL);
  for(int i = 0; i < nrepeat; i++)
    basic_kernel_ab_p_ab(a_basic, b_basic);
  gettimeofday(&end,NULL);
  time = 1.0*(end.tv_sec-begin.tv_sec) + 1.0e-6*(end.tv_usec-begin.tv_usec);
  // Calculate time and flops.
  printf("Base implementation time( %g s )\n", time );
  printf("Base implementation flops( %g GFlops )\n\n", (flops_base = 1.e-9*num_ops/time, flops_base) );
  printf("Floating point performance ratio ( %g \%)\n\n", flops_chylos/flops_base*100.0);

  // Test for numerical match
  double max_abs_diff(0.0);
  double average_abs_diff(0.0);
  for(size_t index=0; index < N; index++){
    double temp_abs_diff( fabs(a[index] - a_basic[index]) );
    average_abs_diff += temp_abs_diff;
    if( temp_abs_diff > max_abs_diff  )
      max_abs_diff = temp_abs_diff;
  }
  printf("Maximum error absolute difference( %g )\n", max_abs_diff);
  printf("Average error absolute difference( %g )\n\n\n", average_abs_diff/N);
}


  Kokkos::finalize();

  return 0;
}
