CXX = nvcc
CPPFLAGS = -g -std=c++17
PROGS = built/old/matmul \
	built/old/sync_vs_overlap \
	built/old/playground \
	built/old/sync_vs_overlap_custom_diag \
	built/old/sync_vs_overlap_diag \
	built/old/sync_vs_overlap_simple \
	built/old/sync_vs_overlap_streams \
	built/old/sync_vs_overlap_one_calc_per_iter_no_cublas \
	built/old/sync_vs_overlap_one_iter \
	built/old/sync_vs_overlap_one_iter_add \
	built/old/sync_vs_overlap_heavy_add \
	built/old/sync_vs_overlap_heavy_add_no_data_race \
	built/old/sync_vs_overlap_heavy_add_diff_issue_order \
	built/sync_vs_overlap_heavy_add_cmp

all: $(PROGS)

built/%: cpp/%.cu
	$(CXX) -o $@ $^ -lcublas

clean:
	rm -rf *.o *~ *.dSYM $(PROGS)
