#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

std::vector<std::pair<size_t, size_t>> merge_intervals(
    std::vector<std::pair<size_t, size_t>> intervals) {
  std::vector<std::pair<size_t, size_t>> merged_intervals;
  if (intervals.empty()) { return merged_intervals; }
  std::sort(intervals.begin(), intervals.end());
  size_t start = intervals[0].first;
  size_t end = intervals[0].second;
  assert(end >= start);
  for (size_t i = 1; i < intervals.size(); i++) {
    assert(intervals[i].second >= intervals[i].first);
    if (intervals[i].first <= end) {
      end = std::max(end, intervals[i].second);
    } else {
      merged_intervals.push_back({start, end});
      start = intervals[i].first;
      end = intervals[i].second;
    }
  }
  merged_intervals.push_back({start, end});
  return merged_intervals;
}
