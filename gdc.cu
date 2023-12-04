#include "gdc.h"
#include "FH.h"
#include "verb.h"

// Main function
int main(int argc, char *argv[]) {
    // Check for correct argument count
    if (argc != 5 && argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <verb> <file1> <file2> <output_file>" << std::endl;
        std::cerr << "Verb: (add|mul|div|exp)" << std::endl; 
        return 1;
    }

    std::string verb = std::string(argv[1]);
    if (verb != "mul" && verb != "add" && verb != "div" && verb != "exp") {
        std::cerr << "Verb: (add|mul|div|exp)" << std::endl; 
        std::cerr << "Invalid verbs" << std::endl;
    }


    if (verb == "add") {
        Add add(argv[2], argv[3], argv[4]);
        add.dispatch();
    } else
    if (verb == "exp") {
        Exp exp1(argv[2], argv[3]);
        exp1.dispatch();
    } else
    if (verb == "mul") {
        Mul mul(argv[2], argv[3], argv[4]);
        mul.dispatch();
    } else
    if (verb == "div") {
        Div div(argv[2], argv[3], argv[4]);
        div.dispatch();
    } /*else
    if (verb == "sin") {
        Exp exp1(argv[2], argv[3]);
        exp1.dispatch();
    } else
    if (verb == "cos") {
        Exp exp1(argv[2], argv[3]);
        exp1.dispatch();
    } else
    if (verb == "tan") {
        Exp exp1(argv[2], argv[3]);
        exp1.dispatch();
    }*/

    return 0;
}
