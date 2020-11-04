LIB_DIR = lib

default: cec20

cec20: setup.py cec20.pyx $(LIB_DIR)/cec20.a
	python setup.py build_ext --inplace && rm -f cec20.c && rm -Rf build

$(LIB_DIR)/cec20.a:
	make -C $(LIB_DIR) libcec20.a

clean:
	rm *.so