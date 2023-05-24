#include "rt.h"
#include "blast.h"
#include <CL/opencl.h>
#include <math.h>
#include <malloc.h>

// because fpp and access enums are used to index arrays they must be compact
// with exact ordering:

static_assert(blast_fpp16 == 0 && blast_fpp32 == 1 && blast_fpp64 == 2, "order");

const char* blast_fpp_names[3] = {"fp16", "fp32", "fp64"};

const int blast_fpp_bytes[3] = {
    (int)sizeof(fp16_t), (int)sizeof(fp32_t), (int)sizeof(fp64_t)
};

static_assert(blast_access_read  == 0, "order");
static_assert(blast_access_write == 1, "order");
static_assert(blast_access_rw    == 2, "order");

static int blast_alloc_access_to_ocl[] = {
    ocl_allocate_read,
    ocl_allocate_write,
    ocl_allocate_rw
};

static int blast_map_access_to_ocl[] = {
    ocl_map_read,
    ocl_map_write,
    ocl_map_rw
};

static blast_memory_t blast_allocate(blast_t* b, int access, int64_t bytes) {
    blast_memory_t gm;
    gm.m = null;
    gm.b = b;
    gm.s = bytes;
    gm.h = ocl.allocate(b->c, blast_alloc_access_to_ocl[access], bytes);
//  traceln("%p: %p", bm->h, bm->m);
    return gm;
}

static void blast_deallocate(blast_memory_t* bm) {
//  traceln("%p: %p", bm->h, bm->m);
    ocl.deallocate((ocl_memory_t)bm->h);
    memset(bm, 0, sizeof(bm));
}

static void* blast_map(blast_memory_t* bm, int access, int64_t offset,
        int64_t bytes) {
    bm->m = ocl.map(bm->b->c, blast_map_access_to_ocl[access],
        (ocl_memory_t)bm->h, offset, bytes);
//  traceln("%p: %p", bm->h, bm->m);
    return bm->m;
}

static void blast_unmap(blast_memory_t* bm) {
//  traceln("%p: %p", bm->h, bm->m);
    ocl.unmap(bm->b->c, (ocl_memory_t)bm->h, bm->m);
    bm->m = null;
}

// Think about what is known in at compiler time for Parallel Reduction
// (e.g. sum of vector elements).
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf


// TODO: there are permutations not addressed here:
// s0 == 1  v1 s1 != 1 (both can be addressed wiht single kernel with
// s0 != 1  v1 s1 == 1  swapped arguments)
// and same for offsets because address + offset still take 1 to 2 cpu
// cycles (int32_t or int64_t)
// all in all, there are 8 combinations with only simplest two implemented
// at the moment. Other can be easily added on AS NEEDED basis for later
// optimization.
// The main goal of blast is to implement gemv(fp16_t) for huge LLM GPT
// and where dot() optimizations may turn to be irrelevant and better
// handled by AVX2/AVX512.

static void blast_dot_compact(int64_t groups, int64_t items,
        blast_memory_t* v0, blast_memory_t* v1, blast_memory_t* r, int fpp) {
    blast_t* b = v0->b;
    ocl_context_t* c = b->c;
    ocl_arg_t args[] = {
        {&v0->h, sizeof(ocl_memory_t)},
        {&v1->h, sizeof(ocl_memory_t)},
        {&r->h,  sizeof(ocl_memory_t)}
    };
    double user = ocl.is_profiling(c) ? seconds() : 0;
    ocl_event_t e = ocl.enqueue_range_kernel(c,
        b->dot_c[fpp], groups, items, countof(args), args);
    user = ocl.is_profiling(c) ? (seconds() - user) : 0;
    if (ocl.is_profiling(c)) {
        ocl_profiling_t* p = ocl.profile_add(c, e);
        p->user = user;
        p->count = groups * items;
        p->fops = 1;
    }
    ocl.release_event(e);
}

static void blast_dot_strided(int64_t groups, int64_t items,
        blast_memory_t* v0, int64_t o0, int64_t s0,
        blast_memory_t* v1, int64_t o1, int64_t s1,
        blast_memory_t* r,  int fpp) {
    blast_t* b = v0->b;
    ocl_context_t* c = b->c;
    ocl_arg_t args[] = {
        {&v0->h, sizeof(ocl_memory_t)},
        {&o0,    sizeof(int32_t)},
        {&s0,    sizeof(int32_t)},
        {&v1->h, sizeof(ocl_memory_t)},
        {&o1,    sizeof(int32_t)},
        {&s1,    sizeof(int32_t)},
        {&r->h,  sizeof(ocl_memory_t)}
    };
    double user = ocl.is_profiling(c) ? seconds() : 0;
    ocl_event_t e = ocl.enqueue_range_kernel(c, b->dot_os[fpp],
        groups, items, countof(args), args);
    user = ocl.is_profiling(c) ? (seconds() - user) : 0;
    if (ocl.is_profiling(c)) {
        ocl_profiling_t* p = ocl.profile_add(c, e);
        p->user = user;
        p->count = groups * items;
        p->fops = 1;
        p->i32ops = 4;
    }
    ocl.release_event(e);
}

static fp64_t read_1xfp_from_memory(blast_memory_t* m, int fpp) {
    fp64_t v = 0;
    void* a = blast.map(m, blast_access_read, 0, blast_fpp_bytes[fpp]);
    switch (fpp) {
        case blast_fpp16: v = fp16to32(*(fp16_t*)a); break;
        case blast_fpp32: v = *(fp32_t*)a; break;
        case blast_fpp64: v = *(fp64_t*)a; break;
        default: fatal_if("fpp", "%d", fpp); break;
    }
    blast.unmap(m);
    return v;
}

static fp64_t sum_and_finish(blast_memory_t* v, int64_t items, int64_t groups, int fpp) {
    blast_t* b = v->b;
    ocl_context_t* c = b->c;
    fp64_t sum = 0;
    int64_t ne = items * groups; // number of elements
    if (ne == 1) {
        ocl.finish(c);
        sum = read_1xfp_from_memory(v, fpp);
    } else {
        int64_t n = ne;
        int64_t m = n / 2;
        int64_t bytes = ne * blast_fpp_bytes[fpp] / 2; // odd "ne" truncated
        blast_memory_t  s = blast.allocate(v->b, blast_access_read, bytes);
        blast_memory_t* v0 = v;
        blast_memory_t* v1 = &s;
        const int64_t max_items  = ocl.devices[c->ix].max_items[0];
        while (m >= 1) {
            if (m < max_items) {
                groups = 1; items = m;
            } else if (groups > 1 && groups % 2 == 0) {
                groups >>= 1;
            } else if (items > 1 && items % 2 == 0) {
                items  >>= 1;
            } else {
                assert(false);
            }
            assertion(groups * items == m);
            ocl_arg_t args[] = {
                {&v0->h,  sizeof(ocl_memory_t)},
                {&v1->h,  sizeof(ocl_memory_t)}
            };
            ocl_kernel_t k = n % 2 == 0 ? b->sum_even[fpp] : b->sum_odd[fpp];
            double user = ocl.is_profiling(c) ? seconds() : 0;
            ocl_event_t e = ocl.enqueue_range_kernel(c, k, groups, items,
                countof(args), args);
            user = ocl.is_profiling(c) ? (seconds() - user) : 0;
            if (ocl.is_profiling(c)) {
                ocl_profiling_t* p = ocl.profile_add(c, e);
                p->user = user;
                p->count = ne;
                p->fops   = 1;
                p->i32ops = 0;
            }
            ocl.release_event(e);
            blast_memory_t* swap = v0; v0 = v1; v1 = swap;
            n  = m;
            m /= 2;
        }
        ocl.finish(c); // same as waiting for chain of events
        sum = read_1xfp_from_memory(v0, fpp);
        blast.deallocate(&s);
    }
    return sum;
}

static fp64_t blast_dot(
        blast_memory_t* v0, int64_t o0, int64_t s0,
        blast_memory_t* v1, int64_t o1, int64_t s1, int64_t n,
        int fpp) { // blast_fpp16, blast_fpp32, blast_fpp64
    fatal_if(v0->b != v1->b, "foreign vectors");
    fatal_if(fpp < blast_fpp16 || blast_fpp64 < fpp, "fpp: %d", fpp);
    blast_t* b = v0->b;
    ocl_context_t* c = b->c;
    fp64_t s = 0;
    const int64_t max_groups = ocl.devices[c->ix].max_groups;
    const int64_t max_items  = ocl.devices[c->ix].max_items[0];
    // sum relies on max_* being power of 2
    assert((max_items  & (max_items  - 1)) == 0);
    assert((max_groups & (max_groups - 1)) == 0);
    if (ocl.is_profiling(c)) {
        c->ov->profiling_count = 0;
    }
    size_t bytes = blast_fpp_bytes[fpp];
    while (n > 0) {
        int64_t groups = min((n + max_items - 1) / max_items, max_groups);
        assertion(n >= (groups - 1) * max_items);
        int64_t ne = groups == 1 ? n : groups * max_items;
        if (groups > 1 && ne > n) { groups--; ne -= max_items; }
        int64_t items = ne / groups;
        assertion(items > 0 && groups > 0 && items * groups <= n);
        assertion(ne == groups * items);
        blast_memory_t r = blast.allocate(b, blast_access_read, ne * bytes);
        if (o0 == 0 && s0 == 1 && o1 == 0 && s1 == 1) {
            blast_dot_compact(groups, items, v0, v1, &r, fpp);
        } else {
//          traceln("offsets: %8lld %8lld strides: %lld %lld ne: %8lld", o0, o1, s0, s1, ne);
            blast_dot_strided(groups, items, v0, o0, s0, v1, o1, s1, &r, fpp);
        }
        s += sum_and_finish(&r, items, groups, fpp);
        blast.deallocate(&r);
        n  -= ne;
        o0 += ne * s0;
        o1 += ne * s1;
    }
    if (ocl.is_profiling(c) && c->ov->profiling_count) {
        ocl_profiling_t* p = &c->ov->profiling[0];
        ocl.profile(&p[0]);
        for (int i = 1; i < c->ov->profiling_count; i++) {
            ocl.profile(&p[i]);
            p[0].time   += p[i].time;
            p[0].user   += p[i].user;
            p[0].gflops += p[i].gflops;
            p[0].i32ops += p[i].i64ops;
            p[0].i64ops += p[i].i64ops;
        }
        p->gflops /= c->ov->profiling_count;
        p->i32ops /= c->ov->profiling_count;
        p->i64ops /= c->ov->profiling_count;
    }
    return s;
}

static fp64_t blast_dot_fp16(
        blast_memory_t* v0, int64_t o0, int64_t s0,
        blast_memory_t* v1, int64_t o1, int64_t s1, int64_t n) {
    return blast_dot(v0, o0, s0, v1, o1, s1, n, blast_fpp16);
}

static fp64_t blast_dot_fp32(
        blast_memory_t* v0, int64_t o0, int64_t s0,
        blast_memory_t* v1, int64_t o1, int64_t s1, int64_t n) {
    return blast_dot(v0, o0, s0, v1, o1, s1, n, blast_fpp32);
}

static fp64_t blast_dot_fp64(
        blast_memory_t* v0, int64_t o0, int64_t s0,
        blast_memory_t* v1, int64_t o1, int64_t s1, int64_t n) {
    return blast_dot(v0, o0, s0, v1, o1, s1, n, blast_fpp64);
}

static const char* blast_program_options(blast_t* b, int fpp) {
    static const char* type_t[] = {"half", "float", "double"};
    static const char* suffix[] = {"fp16", "fp32", "fp64"};
    const char* fp_t = type_t[fpp];
    // see https://man.opencl.org/clBuildProgram.html
    const ocl_device_t* d = &ocl.devices[b->c->ix];
    static char options[4096];
    char* p = options;
    #pragma push_macro("append")
    #define append(...) do {                                             \
        intptr_t k = options + countof(options) - p - 1;                 \
        fatal_if(k <= 0, "options[%d] overflow", (int)countof(options)); \
        p += snprintf(p, k, "" __VA_ARGS__);                             \
    } while (0)
    append("-D fp16_t=half -D fp32_t=float -D fp64_t=double ");
    append("-D int32_t=int -D int64_t=long ");
    append("-cl-std=CL%d.%d ", d->c_version_major, d->c_version_minor);
    append("-D fp_t=%s -D vec4=%s4 -D vec8=%s8 -D vec16=%s16 -D suffix=%s %s ",
           fp_t, fp_t,fp_t, fp_t, suffix[fpp],
          (fpp == blast_fpp16 ? "-D fp16_surrogate" : ""));
    #pragma pop_macro("append")
    *p = 0;
//  traceln("options: %s", options);
    return options;
}

static ocl_program_t blast_compile(blast_t* b, int fpp,
        const void* code, int bytes) {
//  traceln("\nfpp: %s\n%*.*s\n\n", blast_fpp_names[fpp], bytes, bytes, code);
    const char* opts = blast_program_options(b, fpp);
    return ocl.compile_program(b->c, code, bytes, opts);
}

static void blast_init(blast_t* b, ocl_context_t* c) {
    b->c = c;
    ocl_device_t* d = &ocl.devices[b->c->ix];
    void* code = null;
    int64_t bytes64 = 0;
    int r = memmap_resource("blast_cl", &code, &bytes64);
    fatal_if(r != 0 || code == null || bytes64 == 0, "blast.cl in blast.rc?");
    fatal_if(bytes64 > INT_MAX, "blast.cl %lld bytes", bytes64);
    int bytes = (int)bytes64;
    const bool has_fp16 = (d->fp_config & ocl_fp16) != 0;
    const bool has_fp64 =  d->double_fp_config != 0;
    ocl_program_t p[3] = {
        has_fp16 ? blast_compile(b, blast_fpp16, code, bytes) : null,
        blast_compile(b, blast_fpp32, code, bytes),
        has_fp64 ? blast_compile(b, blast_fpp64, code, bytes) : null
    };
    static const char* sum_odd[]     = {"sum_odd_fp16",     "sum_odd_fp32",     "sum_odd_fp64"};
    static const char* sum_odd_os[]  = {"sum_odd_os_fp16",  "sum_odd_os_fp32",  "sum_odd_os_fp64"};
    static const char* sum_even[]    = {"sum_even_fp16",    "sum_even_fp32",    "sum_even_fp64"};
    static const char* sum_even_os[] = {"sum_even_os_fp16", "sum_even_os_fp32", "sum_even_os_fp64"};
    static const char* dot[]         = {"dot_fp16",         "dot_fp32",         "dot_fp64"};
    static const char* dot_os[]      = {"dot_os_fp16",      "dot_os_fp32",      "dot_os_fp64"};
    static const char* gemv[]        = {"gemv_fp16",        "gemv_fp32",        "gemv_fp64"};
    static const char* gemv_os[]     = {"gemv_os_fp16",     "gemv_os_fp32",     "gemv_os_fp64"};
    for (int fp = blast_fpp16; fp <= blast_fpp64; fp++) {
        if (p[fp] != null) {
            b->sum_odd[fp]     = ocl.create_kernel(p[fp], sum_odd[fp]);
            b->sum_odd_os[fp]  = ocl.create_kernel(p[fp], sum_odd_os[fp]);
            b->sum_even[fp]    = ocl.create_kernel(p[fp], sum_even[fp]);
            b->sum_even_os[fp] = ocl.create_kernel(p[fp], sum_even_os[fp]);
            b->dot_c[fp]       = ocl.create_kernel(p[fp], dot[fp]);
            b->dot_os[fp]      = ocl.create_kernel(p[fp], dot_os[fp]);
            b->gemv_c[fp]      = ocl.create_kernel(p[fp], gemv[fp]);
            b->gemv_os[fp]     = ocl.create_kernel(p[fp], gemv_os[fp]);
            ocl.release_program(p[fp]);
            switch (fp) {
                case blast_fpp16: b->dot[fp] = blast_dot_fp16; break;
                case blast_fpp32: b->dot[fp] = blast_dot_fp32; break;
                case blast_fpp64: b->dot[fp] = blast_dot_fp64; break;
                default: fatal_if("never");
            }
        }
    }
}

static void blast_fini(blast_t* b) {
    ocl_device_t* d = &ocl.devices[b->c->ix];
    // all known GPU support at least fp32_t but many do not support
    // fp16_t and/or fp64_t
    int from = (d->fp_config & ocl_fp16) != 0 ? blast_fpp16 : blast_fpp32;
    int to   =  d->double_fp_config != 0 ? blast_fpp64 : blast_fpp32;
    for (int fp = from; fp <= to; fp++) {
        ocl.release_kernel(b->sum_odd[fp]);
        ocl.release_kernel(b->sum_odd_os[fp]);
        ocl.release_kernel(b->sum_even[fp]);
        ocl.release_kernel(b->sum_even_os[fp]);
        ocl.release_kernel(b->dot_c[fp]);
        ocl.release_kernel(b->dot_os[fp]);
        ocl.release_kernel(b->gemv_c[fp]);
        ocl.release_kernel(b->gemv_os[fp]);
    }
}

blast_if blast = {
    .init       = blast_init,
    .allocate   = blast_allocate,
    .deallocate = blast_deallocate,
    .map        = blast_map,
    .unmap      = blast_unmap,
    .fini       = blast_fini
};
