
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64

NVCCFLAGS	:= -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math

double:	main.cu Makefile
	nvcc main.cu -o main $(INC) $(NVCCFLAGS) $(LIB) -DPREC=double

single:	main.cu Makefile
	nvcc main.cu -o main $(INC) $(NVCCFLAGS) $(LIB) -DPREC=float

clean:
	rm -f main
