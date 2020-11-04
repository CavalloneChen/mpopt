CC = gcc

default: libcec17.a

libcec17.a: cec17.o
	ar rcs $@ $^

cec17.o: cec17.c cec17.h
	$(CC) -c $< -fPIC

clean:
	rm *.o *.a
