#include "rt.h"
#include "ocl.h"

enum { KB = 1024, MB = 1024 * KB, GB = 1024 * MB };

enum { N = 1024 * 1024 }; // groups:items NVIDIA 1024:1024 Intel GPU  256:256

// host side arrays (compare time)
static float X[N];
static float Y[N];
static float Z[N];

static double avg_time;
static double avg_user;
static double avg_host;
static double avg_gflops;

static void x_add_y(ocl_context_t* c, ocl_kernel_t k,
                    ocl_memory_t mx, ocl_memory_t my,
                    ocl_memory_t mz, int64_t n, bool verbose) {
    {   // initialize pinned memory:
        float* x = ocl.map(c, ocl_map_write, mx, 0, n * sizeof(float));
        float* y = ocl.map(c, ocl_map_write, my, 0, n * sizeof(float));
        // two input vectors
        for (int32_t i = 0; i < n; i++) { x[i] = (float)i; y[i] = (float)(n - i); }
        for (int32_t i = 0; i < n; i++) { X[i] = x[i]; Y[i] = y[i]; }
        ocl.unmap(c, mx, x);
        ocl.unmap(c, my, y);
    }
    ocl_arg_t args[] =
        {{&mx, sizeof(ocl_memory_t)},
         {&my, sizeof(ocl_memory_t)},
         {&mz, sizeof(ocl_memory_t)}
    };
//  int64_t max_groups = ocl.devices[c->ix].max_groups;
    int64_t max_items  = ocl.devices[c->ix].max_items[0];
    int64_t groups = (n + max_items - 1) / max_items;
    int64_t items  = (n + groups - 1) / groups;
    assert(groups * items == n);
    if (ocl.is_profiling(c)) { c->ov->profiling_count = 0; }
    double time = seconds();
    ocl_event_t done = ocl.enqueue_range_kernel(c, k, groups, items,
        countof(args), args);
    ocl_profiling_t* p = ocl.is_profiling(c) ? ocl.profile_add(c, done) : null;
    // flush() and finish() are unnecessary because ocl.wait(done)
//  ocl.flush(c);
//  ocl.finish(c);
    ocl.wait(&done, 1);
    time = seconds() - time;
    if (p != null) {
        p->user = time;
        p->count = n; // N kernel invocations
        p->fops = 1; //  1 floating operation each
        ocl.profile(p); // collect profiling info and calculate derived values
    }
    ocl.release_event(done); // client's responsibility
    float* z = (float*)ocl.map(c, ocl_map_read, mz, 0, n * sizeof(float));
    double host = 0;
    if (p != null) {
        // measure the same addition of N numbers on CPU
        // L1: 640KB L2: 10MB L3: 24MB guessing L3 may be 4 associative
        int64_t bytes = 24 * 4 * MB;
        byte_t* flush_caches = (byte_t*)malloc(bytes);
        for (int i = 0; i < bytes; i++) { flush_caches[i] = (byte_t)i; }
        host = seconds();
        for (int32_t i = 0; i < n; i++) { Z[i] = X[i] + Y[i]; }
        host = seconds() - host;
        // prevent compiler from optimizing away the above loop
        for (int32_t i = 0; i < n; i++) {
            fatal_if(Z[i] != z[i], "%.1f + %.1f = %.1f\n", X[i], Y[i], z[i]);
        }
        free(flush_caches);
    } else { // just verify result:
        for (int32_t i = 0; i < n; i++) {
            fatal_if(X[i] + Y[i] != z[i], "%.1f + %.1f = %.1f instead of %.1f\n",
                X[i], Y[i], z[i], Z[i]);
        }
    }
    ocl.unmap(c, mz, z);
    if (p != null) {
        if (verbose) {
            traceln("kernel: %6.3f user: %8.3f host: %7.3f (microsec) GFlops: %6.3f",
                    p->time * USEC_IN_SEC, p->user * USEC_IN_SEC, host * USEC_IN_SEC,
                    p->gflops);
        }
        avg_time += p->time;
        avg_user += p->user;
        avg_host += host;
        avg_gflops += p->gflops;
    }
}

#define kernel_name "x_add_y"

static int test(ocl_context_t* c, int64_t n) {
    int result = 0;
    static const char* code =
    "__kernel void " kernel_name "(__global const float* x, "
    "                              __global const float* y, "
    "                              __global float* z) {\n"
    "    int i = get_global_id(0);\n"
    "    z[i] = x[i] + y[i];\n"
    "}\n";
    ocl_program_t p = ocl.compile_program(c, code, strlen(code), null);
    ocl_kernel_t k = ocl.create_kernel(p, kernel_name);
    ocl_memory_t mx = ocl.allocate(c, ocl_allocate_write, n * sizeof(float));
    ocl_memory_t my = ocl.allocate(c, ocl_allocate_write, n * sizeof(float));
    ocl_memory_t mz = ocl.allocate(c, ocl_allocate_read,  n * sizeof(float));
    x_add_y(c, k, mx, my, mz, n, true);
    enum { M = 128 }; // measurements
    for (int i = 0; i < M; i++) {
        x_add_y(c, k, mx, my, mz, n, false);
    }
    if (ocl.is_profiling(c)) {
        avg_time /= M;
        avg_user /= M;
        avg_host /= M;
        avg_gflops /= M;
        traceln("average of %d runs for n: %d", M, n);
        traceln("kernel: %6.3f user: %8.3f host: %7.3f (microsec) GFlops: %6.3f",
                avg_time * USEC_IN_SEC, avg_user * USEC_IN_SEC, avg_host * USEC_IN_SEC,
                avg_gflops);
    }
    // NVIDIA GeForce RTX 3080 Laptop GPU
    // average of 128 runs for n: 1048576
    // kernel: 39.767 user: 1321.327 host: 643.752 (microsec) GFlops: 26.868
    //
    // Intel(R) UHD Graphics
    // average of 128 runs for n: 65536
    // kernel: 14.861 user:  507.260 host:  47.377 (microsec) GFlops:  4.853
    ocl.deallocate(mx); // must be dealloca() before dispose_command_queue()
    ocl.deallocate(my);
    ocl.deallocate(mz);
    ocl.release_kernel(k);
    ocl.release_program(p);
    return result;
}

int32_t main(int32_t argc, const char* argv[]) {
    (void)argc; (void)argv;
    int result = 0;
    ocl.init();
    for (int cycles = 2; cycles > 0; cycles--) {
        for (int i = 0; i < ocl.count; i++) {
            ocl_device_t* d = &ocl.devices[i];
            int64_t n = min(N, d->max_groups * d->max_items[0]);
            ocl_context_t c = ocl.open(i, null);
            traceln("%s\n", ocl.devices[i].name);
            result = test(&c, n);
            traceln("test: %s\n", result == 0 ? "OK" : "FAILED");
            ocl.close(&c);
        }
    }
    ocl_profiling_t p[4096];
    ocl_override_t ov = {
        .max_groups = 0,
        .max_items = 0,
        .profiling = p,
        .max_profiling_count = countof(p),
        .profiling_count = 0
    };
    // profiling measurement:
    for (int i = 0; i < ocl.count; i++) {
        ocl_device_t* d = &ocl.devices[i];
        int64_t n = min(N, d->max_groups * d->max_items[0]);
        ocl_context_t c = ocl.open(i, &ov);
        traceln("%s", ocl.devices[i].name);
        result = test(&c, n);
        traceln("test: %s\n", result == 0 ? "OK" : "FAILED");
        ocl.close(&c);
    }
    return result;
}
