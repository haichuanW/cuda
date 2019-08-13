objects = matrix_transpose

edit:$(objects)

matrix_transpose:matrix_transpose.cu
	nvcc -o matrix_transpose matrix_transpose.cu

.PHONY:clean
clean:$(objects)
		rm $(objects)

