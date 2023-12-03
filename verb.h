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
	FH * host_A_file = nullptr;
	FH * host_B_file = nullptr;
    std::ofstream *outputFilePtr = nullptr;
	int i = _CHUNK_SIZE;
protected:
	size_t A_cols, A_rows, B_cols, B_rows;
	float *device_A, *device_B, *device_C;
public:
	Verb(std::string file1, std::string file2, std::string outputFile);
	Verb(std::string file1, std::string outputFile);
	virtual void execute() {};

	void dispatch();
};

class Add : public Verb {
	using Verb::Verb;
	void execute () override;
};

class Exp : public Verb {
	using Verb::Verb;
	void execute() override;

};

#endif