

.PHONY: compile


compile:
	rm -rf ./generated/
	sbt "test:runMain Picorv32.testMain"



test: testbench.vvp ../firmware/firmware.hex
	vvp -N $<

test_vcd: testbench.vvp ../firmware/firmware.hex
	nnp -N $< +vcd +trace +noerror

testbench.vvp:
	iverilog -o $@ $(subst C,-DCOMPRESSED_ISA,$(COMPRESSED_ISA)) tb/picorv32_tb.v tb/mem.v generated/Picorv32.v

