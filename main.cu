#include <cstdio>
#include <cstring>
#include <cassert>
#include <vector>

#include "fixnum/warp_fixnum.cu"
#include "array/fixnum_array.h"
#include "functions/modexp.cu"
#include "functions/multi_modexp.cu"
#include "modnum/modnum_monty_redc.cu"
#include "modnum/modnum_monty_cios.cu"

using namespace std;
using namespace cuFIXNUM;

const unsigned int bytes_per_elem = 128;
const unsigned int io_bytes_per_elem = 96;

__constant__
const uint8_t non_residue_bytes[bytes_per_elem] = {13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

__constant__
const uint8_t mnt4_modulus_bytes[bytes_per_elem] = {1,128,94,36,222,99,144,94,159,17,221,44,82,84,157,227,240,37,196,154,113,16,136,99,164,84,114,118,233,204,90,104,56,126,83,203,165,13,15,184,157,5,24,242,118,231,23,177,157,247,90,161,217,36,209,153,141,237,160,232,37,185,253,7,115,216,151,108,249,232,183,94,237,175,143,91,80,151,249,183,173,205,226,238,34,144,34,16,17,196,146,45,198,196,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

template <typename fixnum>
class GpuFq
{
  typedef modnum_monty_cios<fixnum> modnum;
private:
  fixnum data_;
  modnum& mod;
public:

  __device__
  GpuFq(const fixnum &data, modnum& mod) : data_(data), mod(mod) {}

  __device__ static GpuFq load(const fixnum &data, modnum& mod) 
  {
    fixnum result;
    mod.to_modnum(result, data);
    return GpuFq(result, mod);
  }

  __device__ static GpuFq loadMM(const fixnum &data, modnum& mod) 
  {
    return GpuFq(data, mod);
  }

  __device__ __forceinline__ void save(fixnum& data) {
    this->mod.from_modnum(data, this->data_);
  }

  __device__ __forceinline__ void saveMM(fixnum& data) {
    data = this->data_;
  }

  __device__ __forceinline__ GpuFq operator*(const GpuFq &other) const
  {
    fixnum result;
    this->mod.mul(result, this->data_, other.data_);
    return GpuFq(result, this->mod);
  }

  __device__ __forceinline__ GpuFq operator+(const GpuFq &other) const
  {
    fixnum result;
    this->mod.add(result, this->data_, other.data_);
    return GpuFq(result, this->mod);
  }

  __device__ __forceinline__ GpuFq operator-(const GpuFq &other) const
  {
    fixnum result;
    this->mod.sub(result, this->data_, other.data_);
    return GpuFq(result, this->mod);
  }

  __device__ __forceinline__ GpuFq squared() const
  {
    fixnum result;
    this->mod.sqr(result, this->data_);
    return GpuFq(result, this->mod);
  }

  __device__ __forceinline__ bool is_zero() const
  {
    return fixnum::is_zero(this->data_);
  }
};

template <typename fixnum>
class GpuFq2 {
  typedef GpuFq<fixnum> GpuFq;

private:
  GpuFq c0, c1;

  GpuFq& non_residue;
public:
  __device__
  GpuFq2(const GpuFq &c0, const GpuFq &c1, GpuFq& non_residue) : c0(c0), c1(c1), non_residue(non_residue) {}

  __device__ __forceinline__ void save(fixnum& c0, fixnum& c1) {
    this->c0.save(c0);
    this->c1.save(c1);
  }

  __device__ __forceinline__ void saveMM(fixnum& c0, fixnum& c1) {
    this->c0.saveMM(c0);
    this->c1.saveMM(c1);
  }

  // __device__ __forceinline__ GpuFq2 operator*(const GpuFq2 &other) const {
  //   GpuFq XX = this->X * other.X;
  //   GpuFq YY = this->Y * other.Y;
  //   GpuFq YX = this->Y * other.X;
  //   GpuFq XY = this->X * other.Y;

  //   return GpuFq2(XX + YY * this->non_residue, YX + XY, this->non_residue);
  // }
  __device__ __forceinline__ GpuFq2 operator*(const GpuFq2 &other) const {
    GpuFq a0_b0 = this->c0 * other.c0;
    GpuFq a1_b1 = this->c1 * other.c1;

    GpuFq a0_plus_a1 = this->c0 + this->c1;
    GpuFq b0_plus_b1 = other.c0 + other.c1;

    GpuFq c = a0_plus_a1 * b0_plus_b1;

    return GpuFq2(a0_b0 + a1_b1 * this->non_residue, c - a0_b0 - a1_b1, this->non_residue);
  }

  __device__ __forceinline__ GpuFq2 operator+(const GpuFq2 &other) const {
    return GpuFq2(this->X + other.X, this->Y + other.Y, this->non_residue);
  }
};


template< typename fixnum >
struct mul_and_convert {
  typedef modnum_monty_cios<fixnum> modnum;
  typedef GpuFq<fixnum> GpuFq;
  typedef GpuFq2<fixnum> GpuFq2;

  __device__ void operator()(fixnum &r0, fixnum &r1, fixnum a0, fixnum a1, fixnum b0, fixnum b1) {
    fixnum n = array_to_fixnum(non_residue_bytes);
    fixnum m = array_to_fixnum(mnt4_modulus_bytes);
    
    modnum mod = modnum(m);
    GpuFq non_residue = GpuFq::load(n, mod);

    GpuFq2 fqA = GpuFq2(GpuFq::load(a0, mod), GpuFq::load(a1, mod), non_residue);
    GpuFq2 fqB = GpuFq2(GpuFq::load(b0, mod), GpuFq::load(b1, mod), non_residue);
    GpuFq2 fqS = fqA * fqB;
    fqS.save(r0, r1);
  }

  __device__ fixnum array_to_fixnum(const uint8_t* arr) {
    return fixnum(((fixnum*)arr)[fixnum::layout::laneIdx()]);
  }
};

template< int fn_bytes, typename fixnum_array >
void print_fixnum_array(fixnum_array* res, int nelts) {
    int lrl = fn_bytes*nelts;
    uint8_t local_results[lrl];
    int ret_nelts;
    for (int i = 0; i < lrl; i++) {
      local_results[i] = 0;
    }
    res->retrieve_all(local_results, fn_bytes*nelts, &ret_nelts);

    for (int i = 0; i < lrl; i++) {
      printf("%i ", local_results[i]);
    }
    printf("\n");
}

template< int fn_bytes, typename fixnum_array >
uint8_t* get_fixnum_array(fixnum_array* res0, fixnum_array* res1, int nelts) {
    int lrl = fn_bytes*nelts;
    uint8_t* local_results0 = new uint8_t[lrl]; 
    uint8_t* local_results1 = new uint8_t[lrl];
    int ret_nelts;
    for (int i = 0; i < lrl; i++) {
      local_results0[i] = 0;
      local_results1[i] = 0;
    }

    res0->retrieve_all(local_results0, fn_bytes*nelts, &ret_nelts);
    res1->retrieve_all(local_results1, fn_bytes*nelts, &ret_nelts);

    uint8_t* local_results = new uint8_t[2 * lrl]; 

    for (int i = 0; i < nelts; i++) {
      mempcpy(local_results + 2 * i * fn_bytes, local_results0 + i * fn_bytes, fn_bytes);
      mempcpy(local_results + 2 * i * fn_bytes + fn_bytes, local_results1 + i * fn_bytes, fn_bytes);
    }

    delete local_results0;
    delete local_results1;
    return local_results;
}


template< int fn_bytes, typename word_fixnum, template <typename> class Func >
uint8_t* compute_product(uint8_t* a, uint8_t* b, int nelts) {
    typedef warp_fixnum<fn_bytes, word_fixnum> fixnum;
    typedef fixnum_array<fixnum> fixnum_array;

    uint8_t *input_a0 = new uint8_t[fn_bytes * nelts];
    uint8_t *input_a1 = new uint8_t[fn_bytes * nelts];

    for (int i = 0; i < nelts; i++) {
      mempcpy(input_a0 + i * fn_bytes, a + 2 * i * fn_bytes, fn_bytes);
      mempcpy(input_a1 + i * fn_bytes, a + 2 * i * fn_bytes + fn_bytes, fn_bytes);
    }

    uint8_t *input_b0 = new uint8_t[fn_bytes * nelts];
    uint8_t *input_b1 = new uint8_t[fn_bytes * nelts];

    for (int i = 0; i < nelts; i++) {
      mempcpy(input_b0 + i * fn_bytes, b + 2 * i * fn_bytes, fn_bytes);
      mempcpy(input_b1 + i * fn_bytes, b + 2 * i * fn_bytes + fn_bytes, fn_bytes);
    }

    fixnum_array *res0, *res1, *in_a0, *in_a1, *in_b0, *in_b1;

    in_a0 = fixnum_array::create(input_a0, fn_bytes * nelts, fn_bytes);
    in_a1 = fixnum_array::create(input_a1, fn_bytes * nelts, fn_bytes);

    in_b0 = fixnum_array::create(input_b0, fn_bytes * nelts, fn_bytes);
    in_b1 = fixnum_array::create(input_b1, fn_bytes * nelts, fn_bytes);
    
    res0 = fixnum_array::create(nelts);
    res1 = fixnum_array::create(nelts);

    fixnum_array::template map<Func>(res0, res1, in_a0, in_a1, in_b0, in_b1);

    uint8_t* v_res = get_fixnum_array<fn_bytes, fixnum_array>(res0, res1, nelts);
    //TODO to do stage 1 field arithmetic, instead of a map, do a reduce

    delete in_a0;
    delete in_a1;
    delete in_b0;
    delete in_b1;
    delete res0;
    delete res1;
    delete[] input_a0;
    delete[] input_a1;
    delete[] input_b0;
    delete[] input_b1;
    return v_res;
}


void read_mnt_fq(uint8_t* dest, FILE* inputs) {
  // the input is montgomery representation x * 2^768 whereas cuda-fixnum expects x * 2^1024 so we shift over by (1024-768)/8 bytes
  fread((void*)(dest), io_bytes_per_elem*sizeof(uint8_t), 1, inputs);
}

void read_mnt_fq2(uint8_t* dest, FILE* inputs) {
  read_mnt_fq(dest, inputs);
  read_mnt_fq(dest + bytes_per_elem, inputs);
}

void write_mnt_fq(uint8_t* src, FILE* outputs) {
  fwrite((void *) src, io_bytes_per_elem * sizeof(uint8_t), 1, outputs);
}

void write_mnt_fq2(uint8_t* src, FILE* outputs) {
  write_mnt_fq(src, outputs);
  write_mnt_fq(src + bytes_per_elem, outputs);
}

void print_array(uint8_t* a) {
  for (int j = 0; j < 128; j++) {
    printf("%x ", ((uint8_t*)(a))[j]);
  }
  printf("\n");
}

int main(int argc, char* argv[]) {
  setbuf(stdout, NULL);

  auto inputs = fopen(argv[2], "r");
  auto outputs = fopen(argv[3], "w");

  size_t n;

  while (true) {
    size_t elts_read = fread((void *) &n, sizeof(size_t), 1, inputs);
    if (elts_read == 0) { break; }

    std::cerr << n << std::endl;

    uint8_t* x0 = new uint8_t[2 * n * bytes_per_elem];
    memset(x0, 0, 2 * n * bytes_per_elem);
    for (size_t i = 0; i < n; ++i) {
      read_mnt_fq2(x0 + 2 * i * bytes_per_elem, inputs);
    }

    uint8_t* x1 = new uint8_t[2 * n * bytes_per_elem];
    memset(x1, 0, 2 * n * bytes_per_elem);
    for (size_t i = 0; i < n; ++i) {
      read_mnt_fq2(x1 + 2 * i * bytes_per_elem, inputs);
    }

    uint8_t* res_x = compute_product<bytes_per_elem, u64_fixnum, mul_and_convert>(x0, x1, n);

    for (size_t i = 0; i < n; ++i) {
      write_mnt_fq2(res_x + 2 * i * bytes_per_elem, outputs);
    }

    delete[] x0;
    delete[] x1;
    delete[] res_x;
  }

  return 0;
}

