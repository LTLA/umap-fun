#include <emscripten/bind.h>

#include "umappp/Umap.hpp"
#include "knncolle/knncolle.hpp"

#include <vector>
#include <cmath>

struct UmapStatus {
    UmapStatus(umappp::Umap<>::Status s) : status(std::move(s)) {}

    umappp::Umap<>::Status status;

    int epoch() const {
        return status.epoch();
    }

    int num_epochs() const {
        return status.num_epochs();
    }
};

UmapStatus initialize_umap(uintptr_t mat, int nr, int nc, int num_neighbors, int num_epochs, double min_dist, bool approximate, uintptr_t Y) {
    std::unique_ptr<knncolle::Base<> > search;
    const double* ptr = reinterpret_cast<const double*>(mat);
    if (approximate) {
        search.reset(new knncolle::AnnoyEuclidean<>(nr, nc, ptr));
    } else {
        search.reset(new knncolle::VpTreeEuclidean<>(nr, nc, ptr));
    }

    umappp::Umap factory;
    factory.set_min_dist(min_dist).set_num_epochs(num_epochs);
    double* embedding = reinterpret_cast<double*>(Y);

    factory.set_num_neighbors(num_neighbors);
    return UmapStatus(factory.initialize(search.get(), 2, embedding));
}
    
bool run_umap(UmapStatus& status, uintptr_t Y) {
    umappp::Umap factory;
    double* ptr = reinterpret_cast<double*>(Y);
    int current = status.epoch();
    const int total = status.num_epochs();
    ++current;
    factory.run(status.status, 2, ptr, current);
    return total == current;
}
    
EMSCRIPTEN_BINDINGS(run_umap) {
    emscripten::function("initialize_umap", &initialize_umap);

    emscripten::function("run_umap", &run_umap);

    emscripten::class_<UmapStatus>("UmapStatus")
        .function("epoch", &UmapStatus::epoch)
        ;
}
