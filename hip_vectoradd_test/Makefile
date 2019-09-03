.PHONY: all clean

all: vector_add

vector_add: vector_add.cpp
	hipcc -g -Wall -Werror -O3 -o vector_add vector_add.cpp

clean:
	rm -f vector_add
