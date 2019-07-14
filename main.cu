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
const uint8_t mnt6_non_residue_bytes[bytes_per_elem] = {11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

__constant__
const uint8_t mnt6_modulus_bytes[bytes_per_elem] = {1,0,0,64,226,118,7,217,79,58,161,15,23,153,160,78,151,87,0,63,188,129,195,214,164,58,153,52,118,249,223,185,54,38,33,41,148,202,235,62,155,169,89,200,40,92,108,178,157,247,90,161,217,36,209,153,141,237,160,232,37,185,253,7,115,216,151,108,249,232,183,94,237,175,143,91,80,151,249,183,173,205,226,238,34,144,34,16,17,196,146,45,198,196,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

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
  //   GpuFq c0_c0 = this->c0 * other.c0;
  //   GpuFq c1_c1 = this->c1 * other.c1;
  //   GpuFq c1_c0 = this->c1 * other.c0;
  //   GpuFq c0_c1 = this->c0 * other.c1;

  //   return GpuFq2(c0_c0 + c1_c1 * this->non_residue, c1_c0 + c0_c1, this->non_residue);
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
    return GpuFq2(this->c0 + other.c0, this->c1 + other.c1, this->non_residue);
  }
};

template <typename fixnum>
class GpuFq3 {
  typedef GpuFq<fixnum> GpuFq;

private:
  GpuFq c0, c1, c2;

  GpuFq& non_residue;
public:
  __device__
  GpuFq3(const GpuFq &c0, const GpuFq &c1, const GpuFq &c2, GpuFq& non_residue) : c0(c0), c1(c1), c2(c2), non_residue(non_residue) {}

  __device__ __forceinline__ void save(fixnum& c0, fixnum& c1, fixnum& c2) {
    this->c0.save(c0);
    this->c1.save(c1);
    this->c2.save(c2);
  }

  __device__ __forceinline__ void saveMM(fixnum& c0, fixnum& c1, fixnum& c2) {
    this->c0.saveMM(c0);
    this->c1.saveMM(c1);
    this->c2.saveMM(c2);
  }

  __device__ __forceinline__ GpuFq3 operator*(const GpuFq3 &other) const {
    GpuFq c0_c0 = this->c0 * other.c0;
    GpuFq c0_c1 = this->c0 * other.c1;
    GpuFq c0_c2 = this->c0 * other.c2;

    GpuFq c1_c0 = this->c1 * other.c0;
    GpuFq c1_c1 = this->c1 * other.c1;
    GpuFq c1_c2 = this->c1 * other.c2;
    
    GpuFq c2_c0 = this->c2 * other.c0;
    GpuFq c2_c1 = this->c2 * other.c1;
    GpuFq c2_c2 = this->c2 * other.c2;

    return GpuFq3(
      c0_c0 + this->non_residue * (c1_c2 + c2_c1),
      c0_c1 + c1_c0 + this->non_residue * c2_c2,
      c0_c2 + c1_c1 + c2_c0,
      this->non_residue
    );
  }
  // __device__ __forceinline__ GpuFq2 operator*(const GpuFq2 &other) const {
  //   GpuFq a0_b0 = this->c0 * other.c0;
  //   GpuFq a1_b1 = this->c1 * other.c1;

  //   GpuFq a0_plus_a1 = this->c0 + this->c1;
  //   GpuFq b0_plus_b1 = other.c0 + other.c1;

  //   GpuFq c = a0_plus_a1 * b0_plus_b1;

  //   return GpuFq2(a0_b0 + a1_b1 * this->non_residue, c - a0_b0 - a1_b1, this->non_residue);
  // }

  __device__ __forceinline__ GpuFq3 operator+(const GpuFq3 &other) const {
    return GpuFq3(this->c0 + other.c0, this->c1 + other.c1, this->c2 + other.c2, this->non_residue);
  }
};


template< typename fixnum >
struct mul_and_convert {
  typedef modnum_monty_cios<fixnum> modnum;
  typedef GpuFq<fixnum> GpuFq;
  typedef GpuFq3<fixnum> GpuFq3;

  __device__ void operator()(fixnum &r0, fixnum &r1, fixnum &r2, fixnum a0, fixnum a1, fixnum a2, fixnum b0, fixnum b1, fixnum b2) {
    fixnum n = array_to_fixnum(mnt6_non_residue_bytes);
    fixnum m = array_to_fixnum(mnt6_modulus_bytes);
    
    modnum mod = modnum(m);
    GpuFq non_residue = GpuFq::load(n, mod);

    GpuFq3 fqA = GpuFq3(GpuFq::load(a0, mod), GpuFq::load(a1, mod), GpuFq::load(a2, mod), non_residue);
    GpuFq3 fqB = GpuFq3(GpuFq::load(b0, mod), GpuFq::load(b1, mod), GpuFq::load(b2, mod), non_residue);
    GpuFq3 fqS = fqA * fqB;
    fqS.save(r0, r1, r2);
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
uint8_t* get_fixnum_array(fixnum_array* res0, fixnum_array* res1, fixnum_array* res2, int nelts) {
    int lrl = fn_bytes*nelts;
    uint8_t* local_results0 = new uint8_t[lrl]; 
    uint8_t* local_results1 = new uint8_t[lrl];
    uint8_t* local_results2 = new uint8_t[lrl];
    int ret_nelts;
    for (int i = 0; i < lrl; i++) {
      local_results0[i] = 0;
      local_results1[i] = 0;
      local_results2[i] = 0;
    }

    res0->retrieve_all(local_results0, fn_bytes*nelts, &ret_nelts);
    res1->retrieve_all(local_results1, fn_bytes*nelts, &ret_nelts);
    res2->retrieve_all(local_results2, fn_bytes*nelts, &ret_nelts);

    uint8_t* local_results = new uint8_t[3 * lrl]; 

    for (int i = 0; i < nelts; i++) {
      mempcpy(local_results + 3 * i * fn_bytes, local_results0 + i * fn_bytes, fn_bytes);
      mempcpy(local_results + 3 * i * fn_bytes + fn_bytes, local_results1 + i * fn_bytes, fn_bytes);
      mempcpy(local_results + 3 * i * fn_bytes + 2 * fn_bytes, local_results2 + i * fn_bytes, fn_bytes);
    }

    delete[] local_results0;
    delete[] local_results1;
    delete[] local_results2;
    return local_results;
}


template< int fn_bytes, typename word_fixnum, template <typename> class Func >
uint8_t* compute_product(uint8_t* a, uint8_t* b, int nelts) {
    typedef warp_fixnum<fn_bytes, word_fixnum> fixnum;
    typedef fixnum_array<fixnum> fixnum_array;

    uint8_t *input_a0 = new uint8_t[fn_bytes * nelts];
    uint8_t *input_a1 = new uint8_t[fn_bytes * nelts];
    uint8_t *input_a2 = new uint8_t[fn_bytes * nelts];

    for (int i = 0; i < nelts; i++) {
      mempcpy(input_a0 + i * fn_bytes, a + 3 * i * fn_bytes, fn_bytes);
      mempcpy(input_a1 + i * fn_bytes, a + 3 * i * fn_bytes + fn_bytes, fn_bytes);
      mempcpy(input_a2 + i * fn_bytes, a + 3 * i * fn_bytes + 2 * fn_bytes, fn_bytes);
    }

    uint8_t *input_b0 = new uint8_t[fn_bytes * nelts];
    uint8_t *input_b1 = new uint8_t[fn_bytes * nelts];
    uint8_t *input_b2 = new uint8_t[fn_bytes * nelts];

    for (int i = 0; i < nelts; i++) {
      mempcpy(input_b0 + i * fn_bytes, b + 3 * i * fn_bytes, fn_bytes);
      mempcpy(input_b1 + i * fn_bytes, b + 3 * i * fn_bytes + fn_bytes, fn_bytes);
      mempcpy(input_b2 + i * fn_bytes, b + 3 * i * fn_bytes + 2 * fn_bytes, fn_bytes);
    }

    fixnum_array *res0, *res1, *res2, *in_a0, *in_a1, *in_a2, *in_b0, *in_b1, *in_b2;

    in_a0 = fixnum_array::create(input_a0, fn_bytes * nelts, fn_bytes);
    in_a1 = fixnum_array::create(input_a1, fn_bytes * nelts, fn_bytes);
    in_a2 = fixnum_array::create(input_a2, fn_bytes * nelts, fn_bytes);

    in_b0 = fixnum_array::create(input_b0, fn_bytes * nelts, fn_bytes);
    in_b1 = fixnum_array::create(input_b1, fn_bytes * nelts, fn_bytes);
    in_b2 = fixnum_array::create(input_b2, fn_bytes * nelts, fn_bytes);
    
    res0 = fixnum_array::create(nelts);
    res1 = fixnum_array::create(nelts);
    res2 = fixnum_array::create(nelts);

    fixnum_array::template map<Func>(res0, res1, res2, in_a0, in_a1, in_a2, in_b0, in_b1, in_b2);

    uint8_t* v_res = get_fixnum_array<fn_bytes, fixnum_array>(res0, res1, res2, nelts);
    //TODO to do stage 1 field arithmetic, instead of a map, do a reduce

    delete in_a0;
    delete in_a1;
    delete in_a2;
    delete in_b0;
    delete in_b1;
    delete in_b2;
    delete res0;
    delete res1;
    delete res2;
    delete[] input_a0;
    delete[] input_a1;
    delete[] input_a2;
    delete[] input_b0;
    delete[] input_b1;
    delete[] input_b2;
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

void read_mnt_fq3(uint8_t* dest, FILE* inputs) {
  read_mnt_fq(dest, inputs);
  read_mnt_fq(dest + bytes_per_elem, inputs);
  read_mnt_fq(dest + 2 * bytes_per_elem, inputs);
}

void write_mnt_fq(uint8_t* src, FILE* outputs) {
  fwrite((void *) src, io_bytes_per_elem * sizeof(uint8_t), 1, outputs);
}

void write_mnt_fq2(uint8_t* src, FILE* outputs) {
  write_mnt_fq(src, outputs);
  write_mnt_fq(src + bytes_per_elem, outputs);
}

void write_mnt_fq3(uint8_t* src, FILE* outputs) {
  write_mnt_fq(src, outputs);
  write_mnt_fq(src + bytes_per_elem, outputs);
  write_mnt_fq(src + 2 * bytes_per_elem, outputs);
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

    uint8_t* x0 = new uint8_t[3 * n * bytes_per_elem];
    memset(x0, 0, 3 * n * bytes_per_elem);
    for (size_t i = 0; i < n; ++i) {
      read_mnt_fq3(x0 + 3 * i * bytes_per_elem, inputs);
    }

    uint8_t* x1 = new uint8_t[3 * n * bytes_per_elem];
    memset(x1, 0, 3 * n * bytes_per_elem);
    for (size_t i = 0; i < n; ++i) {
      read_mnt_fq3(x1 + 3 * i * bytes_per_elem, inputs);
    }

    uint8_t* res_x = compute_product<bytes_per_elem, u64_fixnum, mul_and_convert>(x0, x1, n);

    for (size_t i = 0; i < n; ++i) {
      write_mnt_fq3(res_x + 3 * i * bytes_per_elem, outputs);
    }

    delete[] x0;
    delete[] x1;
    delete[] res_x;
  }

  return 0;
}

