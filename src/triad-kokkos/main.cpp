#include <cstdlib>
#include <iostream>
#include <Kokkos_Core.hpp>

#include "OptionParser.h"
#include "Utility.h"

using namespace std;

void addBenchmarkSpecOptions(OptionParser &op);
void RunBenchmark(OptionParser &op);

int main(int argc, char* argv[]) {
  int ret = 0;

  try {
    OptionParser op;
    op.addOption("verbose", OPT_BOOL, "", "enable verbose output", 'v');
    op.addOption("passes", OPT_INT, "10", "specify number of passes", 'n');

    addBenchmarkSpecOptions(op);

    if (!op.parse(argc, argv)) {
      op.usage();
      return (op.HelpRequested() ? 0 : 1);
    }

    Kokkos::initialize(argc, argv);
    {
      RunBenchmark(op);
    }
    Kokkos::finalize();
  }
  catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    ret = 1;
  }
  catch (...) {
    ret = 1;
  }

  return ret;
}
