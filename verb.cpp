#include "verb.h"

Verb::Verb(std::string file1) {
       _file1 = new FH(file1);
}

Verb::Verb(std::string file1, std::string file2) {
       _file1 = new FH(file1);
       _file2 = new FH(file2);
}

void Verb::dispatch() {
       if (_file2 == nullptr) {
              // 1 file operations
              execute();
       } else {
              // 2 file operations
              execute();
       }
}

void Add::execute() {
       std::cout << "Add" << std::endl;
}