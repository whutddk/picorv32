

.PHONY: compile test test_vcd clean


compile:
	rm -rf ./generated/
	sbt "test:runMain Picorv32.testMain"



test: 
	iverilog -o ./tb/build/testbench.vvp -D RANDOMIZE_REG_INIT tb/picorv32_tb.v tb/mem.v generated/Picorv32.v
	vvp -N ./tb/build/testbench.vvp ../firmware/firmware.hex +RANDOMIZE_REG_INIT

test_vcd:
	iverilog -o ./tb/build/testbench.vvp -D RANDOMIZE_REG_INIT tb/picorv32_tb.v tb/mem.v generated/Picorv32.v
	vvp -N ./tb/build/testbench.vvp ../firmware/firmware.hex +vcd +trace +noerror +RANDOMIZE_REG_INIT

test_diff:
	iverilog -o ./tb/build/testbench.vvp -D RANDOMIZE_REG_INIT -D ORI_DIFF tb/picorv32_tb.v tb/mem.v generated/Picorv32.v ../picorv32.v
	vvp -N ./tb/build/testbench.vvp ../firmware/firmware.hex +vcd +trace +noerror +RANDOMIZE_REG_INIT

clean:
	rm -vrf \
		./tb/build/testbench.vvp ./tb/build/testbench_sp.vvp ./tb/build/testbench_synth.vvp ./tb/build/testbench_ez.vvp \
		./tb/build/testbench_rvf.vvp ./tb/build/testbench_wb.vvp ./tb/build/testbench.vcd ./tb/build/testbench.trace \
		./tb/build/testbench_verilator ./tb/build/testbench_verilator_dir


vcd:
	gtkwave ./tb/build/wave.vcd &

