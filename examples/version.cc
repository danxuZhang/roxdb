#include <iostream>

#include "roxdb/db.h"

int main() {
  std::cout << "rox::DB::GetVersion() = " << rox::DB::GetVersion() << std::endl;
  return 0;
}
