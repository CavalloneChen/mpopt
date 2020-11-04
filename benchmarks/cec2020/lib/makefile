CC = gcc

default: libcec20.a

libcec20.a: cec20.o
	ar rcs $@ $^

cec20.o: cec20.c cec20.h
	$(CC) -c $< -fPIC

clean:
	rm *.o *.a
