#include <math.h>
//#include <cuda.h>
#include <builtin_types.h>
//#include <drvapi_error_string.h>
#include "nvvm.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string>

namespace QDP {


char *loadProgramSource(const char *filename, size_t *size) 
{
    struct stat statbuf;
    FILE *fh;
    char *source = NULL;
    *size = 0;
    fh = fopen(filename, "rb");
    if (fh) {
        stat(filename, &statbuf);
        source = (char *) malloc(statbuf.st_size+1);
        if (source) {
            fread(source, statbuf.st_size, 1, fh);
            source[statbuf.st_size] = 0;
            *size = statbuf.st_size+1;
        }
    }
    else {
        fprintf(stderr, "Error reading file %s\n", filename);
        exit(-1);
    }
    return source;
}

char *generatePTX(const char *ll, size_t size)
{
    nvvmResult result;
    nvvmCU cu;

    result = nvvmInit();
    if (result != NVVM_SUCCESS) {
        fprintf(stderr, "nvvmInit: Failed\n");
	exit(-1);
    }

    result = nvvmCreateCU(&cu);
    if (result != NVVM_SUCCESS) {
        fprintf(stderr, "nvvmCreateCU: Failed\n");
        exit(-1); 
    }

    result = nvvmCUAddModule(cu, ll, size);
    if (result != NVVM_SUCCESS) {
        fprintf(stderr, "nvvmCUAddModule: Failed\n");
        exit(-1);
    }
 
    result = nvvmCompileCU(cu,  0, NULL);
    if (result != NVVM_SUCCESS) {
        fprintf(stderr, "nvvmCompileCU: Failed\n");
        size_t LogSize;
        nvvmGetCompilationLogSize(cu, &LogSize);
        char *Msg = (char*)malloc(LogSize);
        nvvmGetCompilationLog(cu, Msg);
        fprintf(stderr, "%s\n", Msg);
        free(Msg);
        nvvmFini();
        exit(-1);
    }
    
    size_t PTXSize;
    result = nvvmGetCompiledResultSize(cu, &PTXSize);
    if (result != NVVM_SUCCESS) {
        fprintf(stderr, "nvvmGetCompiledResultSize: Failed\n");
        exit(-1);
    }
    
    char *PTX = (char*)malloc(PTXSize);
    result = nvvmGetCompiledResult(cu, PTX);
    if (result != NVVM_SUCCESS) {
        fprintf(stderr, "nvvmGetCompiledResult: Failed\n");
        free(PTX);
        exit(-1);
    }
    
    result = nvvmDestroyCU(&cu);
    if (result != NVVM_SUCCESS) {
      fprintf(stderr, "nvvmDestroyCU: Failed\n");
      free(PTX);
      exit(-1);
    }
    
    result = nvvmFini();
    if (result != NVVM_SUCCESS) {
      fprintf(stderr, "nvvmFini: Failed\n");
      free(PTX);
      exit(-1);
    }
    
    return PTX;
}



std::string nvvm_compile(const char* ll_kernel_fname) 
{
  size_t size = 0;
  char *ll = NULL;

  ll = loadProgramSource( ll_kernel_fname, &size);

  fprintf(stdout, "NVVM IR ll file loaded\n");

  // Use libnvvm to generte PTX
  char *ptx = generatePTX(ll, size);
  fprintf(stdout, "PTX generated:\n");
  fprintf(stdout, "%s\n", ptx);

  std::string ret(ptx);

  free(ll);
  free(ptx);

  return ret;
}

} // namespace QDP
