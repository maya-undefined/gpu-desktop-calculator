#ifndef _VERB_H
#define _VERB_H

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include "gdc.h"
#include "FH.h"

class Verb {
private:
	FH * _file1 = nullptr;
	FH * _file2 = nullptr;
	int i = _CHUNK_SIZE;
public:
	Verb(std::string file1, std::string file2);
	Verb(std::string file1);
	virtual void execute() {};
	void dispatch();
};

class Add : public Verb {
	using Verb::Verb;
	void execute () override;
	// void execute() {
	// 	std::cout << "Add" << std::endl;
	// }
};

#endif