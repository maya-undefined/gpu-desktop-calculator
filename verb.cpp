#include "verb.h"

Verb::Verb(std::string verb, std::string file1) {
       _verb = verb;
       _file1 = new FH(file1);
}

Verb::Verb(const std::string &verb, std::string file1, std::string file2) {
       _verb = verb;
       _file1 = new FH(file1);
       _file2 = new FH(file2);
}

Verb::dispatch() {
       if (_file2 == nullptr) {
              // 1 file operations
              this.execute();
       } else {
              // 2 file operations
              this.execute();
       }
}