LIB_DIR = lib

default: cec17

cec17: setup.py cec17.pyx $(LIB_DIR)/cec17.a
	python setup.py build_ext --inplace && rm -f cec17.c && rm -Rf build

$(LIB_DIR)/cec17.a:
	make -C $(LIB_DIR) libcec17.a

clean:
	rm *.so