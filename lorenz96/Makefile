INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64

NVCCFLAGS	:= -lineinfo -arch=compute_60 --ptxas-options=-v --use_fast_math

double:	main.cu Makefile
	nvcc main.cu -o main $(INC) $(NVCCFLAGS) $(LIB) -DPREC=double -DTYPE=0

single:	main.cu Makefile
	nvcc main.cu -o main $(INC) $(NVCCFLAGS) $(LIB) -DPREC=float -DTYPE=1

half:	main.cu Makefile
	nvcc main.cu -o main $(INC) $(NVCCFLAGS) $(LIB) -DPREC=half -DTYPE=2

clean:
	rm -f main
