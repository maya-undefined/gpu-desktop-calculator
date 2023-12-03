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
public:
	Verb(std::string file1, std::string file2, std::string outputFile);
	Verb(std::string file1, std::string outputFile);
	virtual void execute() {};
	virtual void execute(size_t A_rows, size_t A_cols, size_t B_rows, size_t B_cols, float *device_C, float *device_B, float *device_A ) {};

	void dispatch();
};

class Add : public Verb {
	using Verb::Verb;
	void execute(size_t A_rows, size_t A_cols, size_t B_rows, size_t B_cols, float *device_C, float *device_B, float *device_A ) override;
	void execute () override {};
};

class Exp : public Verb {
	using Verb::Verb;
	void execute() override;
	void execute(size_t A_rows, size_t A_cols, size_t B_rows, size_t B_cols, float *device_C, float *device_B, float *device_A ) override {};

};

#endif