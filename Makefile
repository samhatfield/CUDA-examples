INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64

NVCCFLAGS	:= -lineinfo -arch=compute_60 --ptxas-options=-v --use_fast_math

half:	main.cu Makefile
	nvcc main.cu -o main $(INC) $(NVCCFLAGS) $(LIB)

clean:
	rm -f main
