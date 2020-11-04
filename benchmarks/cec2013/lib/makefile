CC = gcc

default: libcec13.a

libcec13.a: cec13.o
	ar rcs $@ $^

cec13.o: cec13.c cec13.h
	$(CC) -c $< -fPIC

clean:
	rm *.o *.a
