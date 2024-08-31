CXX = nvcc
CPPFLAGS = -g -std=c++17
PROGS = built/sync_vs_overlap built/playground built/sync_vs_overlap_custom_diag \
	built/sync_vs_overlap_diag built/sync_vs_overlap_simple built/sync_vs_overlap_streams \
	built/sync_vs_overlap_one_calc_per_iter_no_cublas built/sync_vs_overlap_one_iter \
	built/sync_vs_overlap_one_iter_add built/sync_vs_overlap_heavy_add \
	built/sync_vs_overlap_heavy_add_no_data_race \
	built/sync_vs_overlap_heavy_add_diff_issue_order \
	built/sync_vs_overlap_heavy_add_cmp \

all: $(PROGS)

built/sync_vs_overlap: sync_vs_overlap.cu
	$(CXX) -o $@ $^ -lcublas

built/playground: playground.cu
	$(CXX) -o $@ $^ -lcublas

built/sync_vs_overlap_custom_diag: sync_vs_overlap_custom_diag.cu
	$(CXX) -o $@ $^ -lcublas

built/sync_vs_overlap_diag: sync_vs_overlap_diag.cu
	$(CXX) -o $@ $^ -lcublas

built/sync_vs_overlap_simple: sync_vs_overlap_simple.cu
	$(CXX) -o $@ $^ -lcublas

built/sync_vs_overlap_streams: sync_vs_overlap_streams.cu
	$(CXX) -o $@ $^ -lcublas

built/sync_vs_overlap_one_calc_per_iter_no_cublas: sync_vs_overlap_one_calc_per_iter_no_cublas.cu
	$(CXX) -o $@ $^

built/sync_vs_overlap_one_iter: sync_vs_overlap_one_iter.cu -lcublas
	$(CXX) -o $@ $^

built/sync_vs_overlap_one_iter_add: sync_vs_overlap_one_iter_add.cu
	$(CXX) -o $@ $^

built/sync_vs_overlap_heavy_add: sync_vs_overlap_heavy_add.cu
	$(CXX) -o $@ $^

built/sync_vs_overlap_heavy_add_no_data_race: sync_vs_overlap_heavy_add_no_data_race.cu
	$(CXX) -o $@ $^

built/sync_vs_overlap_heavy_add_diff_issue_order: sync_vs_overlap_heavy_add_diff_issue_order.cu
	$(CXX) -o $@ $^

built/%: %.cu
	$(CXX) -o $@ $^

clean:
	rm -rf *.o *~ *.dSYM $(PROGS)
