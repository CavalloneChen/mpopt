LIB_DIR = lib

default: cec13

cec13: setup.py cec13.pyx $(LIB_DIR)/cec13.a
	python setup.py build_ext --inplace && rm -f cec13.c && rm -Rf build

$(LIB_DIR)/cec13.a:
	make -C $(LIB_DIR) libcec13.a

clean:
	rm *.so