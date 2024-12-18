.PHONY: test fst clean

VSimTop: 
	rm -rf /${PWD}/build/
	mkdir -p /${PWD}/build/
	verilator -Wno-fatal  \
	--timescale "1 ns / 1 ps" \
	-y /${PWD} \
	--top-module SimTop \
	--trace-fst \
	--cc /${PWD}/SimTop.v  \
	+define+RANDOMIZE_GARBAGE_ASSIGN \
	+define+RANDOMIZE_INVALID_ASSIGN \
	+define+RANDOMIZE_REG_INIT \
	+define+RANDOMIZE_MEM_INIT \
	+define+RANDOMIZE_DELAY=0 \
	+define+USE_POWER_PINS \
	--exe --build \
	/${PWD}/sim_main.cpp  \
	-Mdir ./build/ \
	-j 8

test: VSimTop
	./build/VSimTop -w -l -f ./firmware.hex

fst:
	gtkwave ./build/wave.fst &
