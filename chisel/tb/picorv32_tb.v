`timescale 1 ns / 1 ps


module picorv32_tb (

);


	reg clock;
	reg reset;


    wire    mem_valid;
    wire    mem_ready;
    wire  [31:0] mem_addr;
    wire  [31:0] mem_wdata;
    wire  [3:0]  mem_wstrb;
    wire  [31:0] mem_rdata;


	wire trap;
	wire trace_valid;
	wire [35:0] trace_data;



	wire tests_passed;
	reg [31:0] irq = 0;

	reg [15:0] count_cycle = 0;


	wire        rvfi_valid;
	wire [63:0] rvfi_order;
	wire [31:0] rvfi_insn;
	wire        rvfi_trap;
	wire        rvfi_halt;
	wire        rvfi_intr;
	wire [4:0]  rvfi_rs1_addr;
	wire [4:0]  rvfi_rs2_addr;
	wire [31:0] rvfi_rs1_rdata;
	wire [31:0] rvfi_rs2_rdata;
	wire [4:0]  rvfi_rd_addr;
	wire [31:0] rvfi_rd_wdata;
	wire [31:0] rvfi_pc_rdata;
	wire [31:0] rvfi_pc_wdata;
	wire [31:0] rvfi_mem_addr;
	wire [3:0]  rvfi_mem_rmask;
	wire [3:0]  rvfi_mem_wmask;
	wire [31:0] rvfi_mem_rdata;
	wire [31:0] rvfi_mem_wdata;








    Picorv32 s_picorv32(
        .clock(clock),
        .reset(reset),

        .io_trap(trap),

        .io_mem_valid(mem_valid),
        .io_mem_ready(mem_ready),
        .io_mem_isInstr(),
        .io_mem_addr(mem_addr),
        .io_mem_wdata(mem_wdata),
        .io_mem_wstrb(mem_wstrb),
        .io_mem_rdata(mem_rdata),

        .io_mem_la_read(),
        .io_mem_la_write(),
        .io_mem_la_addr(),
        .io_mem_la_wdata(),
        .io_mem_la_wstrb(),
        .io_pcpi_valid(),

        .io_pcpi_ready(1'b1),
        .io_pcpi_insn(),
        .io_pcpi_rs1(),
        .io_pcpi_rs2(),
        .io_pcpi_wr(1'b0),
        .io_pcpi_rd(32'd0),
        .io_pcpi_waiting(1'b1),

        .io_irq(irq),
        .io_eoi(),

        .io_rvfi_valid(rvfi_valid     ),
        .io_rvfi_order(rvfi_order     ),
        .io_rvfi_insn (rvfi_insn      ),
        .io_rvfi_trap (rvfi_trap      ),
        .io_rvfi_halt (rvfi_halt      ),
        .io_rvfi_intr (rvfi_intr      ),
        .io_rvfi_mode(),
        .io_rvfi_ixl (),
        .io_rvfi_rs1_addr (rvfi_rs1_addr  ),
        .io_rvfi_rs2_addr (rvfi_rs2_addr  ),
        .io_rvfi_rs1_rdata(rvfi_rs1_rdata ),
        .io_rvfi_rs2_rdata(rvfi_rs2_rdata ),
        .io_rvfi_rd_addr  (rvfi_rd_addr   ),
        .io_rvfi_rd_wdata (rvfi_rd_wdata  ),
        .io_rvfi_pc_rdata (rvfi_pc_rdata  ),
        .io_rvfi_pc_wdata (rvfi_pc_wdata  ),
        .io_rvfi_mem_addr (rvfi_mem_addr  ),
        .io_rvfi_mem_rmask(rvfi_mem_rmask ),
        .io_rvfi_mem_wmask(rvfi_mem_wmask ),
        .io_rvfi_mem_rdata(rvfi_mem_rdata ),
        .io_rvfi_mem_wdata(rvfi_mem_wdata ),

        .io_rvfi_csr_mcycle_rmask(),
        .io_rvfi_csr_mcycle_wmask(),
        .io_rvfi_csr_mcycle_rdata(),
        .io_rvfi_csr_mcycle_wdata(),
        .io_rvfi_csr_minstret_rmask(),
        .io_rvfi_csr_minstret_wmask(),
        .io_rvfi_csr_minstret_rdata(),
        .io_rvfi_csr_minstret_wdata(),

        .io_trace_valid(trace_valid),
        .io_trace_bits(trace_data)
    );

    mem s_mem(
        .clock(clock),
        .mem_valid(mem_valid),
        .mem_ready(mem_ready),
        .mem_addr(mem_addr),
        .mem_wdata(mem_wdata),
        .mem_wstrb(mem_wstrb),
        .mem_rdata(mem_rdata),
        .tests_passed(tests_passed)
    );

	picorv32_rvfimon rvfi_monitor (
		.clock          (clk           ),
		.reset          (!resetn       ),
		.rvfi_valid     (rvfi_valid    ),
		.rvfi_order     (rvfi_order    ),
		.rvfi_insn      (rvfi_insn     ),
		.rvfi_trap      (rvfi_trap     ),
		.rvfi_halt      (rvfi_halt     ),
		.rvfi_intr      (rvfi_intr     ),
		.rvfi_rs1_addr  (rvfi_rs1_addr ),
		.rvfi_rs2_addr  (rvfi_rs2_addr ),
		.rvfi_rs1_rdata (rvfi_rs1_rdata),
		.rvfi_rs2_rdata (rvfi_rs2_rdata),
		.rvfi_rd_addr   (rvfi_rd_addr  ),
		.rvfi_rd_wdata  (rvfi_rd_wdata ),
		.rvfi_pc_rdata  (rvfi_pc_rdata ),
		.rvfi_pc_wdata  (rvfi_pc_wdata ),
		.rvfi_mem_addr  (rvfi_mem_addr ),
		.rvfi_mem_rmask (rvfi_mem_rmask),
		.rvfi_mem_wmask (rvfi_mem_wmask),
		.rvfi_mem_rdata (rvfi_mem_rdata),
		.rvfi_mem_wdata (rvfi_mem_wdata)
	);


    initial begin
        clock = 0;
        reset = 1;
    end


	always #5 clock = ~clock;

	initial begin
		repeat (100) @(posedge clock);
		reset <= 0;
	end

	initial begin
		if ($test$plusargs("vcd")) begin
			$dumpfile("./build/wave.vcd"); //生成的vcd文件名称
			$dumpvars(0, picorv32_tb);//tb模块名称
		end
		repeat (1000000) @(posedge clk);
		$display("TIMEOUT");
		$finish;
	end


	integer trace_file;

	initial begin
		if ($test$plusargs("trace")) begin
			trace_file = $fopen("./build/testbench.trace", "w");
			repeat (10) @(posedge clk);
			while (!trap) begin
				@(posedge clk);
				if (trace_valid)
					$fwrite(trace_file, "%x\n", trace_data);
			end
			$fclose(trace_file);
			$display("Finished writing testbench.trace.");
		end
	end



	always @(posedge clk) count_cycle <= ~reset ? count_cycle + 1 : 0;

	always @* begin
		irq = 0;
		irq[4] = &count_cycle[12:0];
		irq[5] = &count_cycle[15:0];
	end













	integer cycle_counter;
	always @(posedge clk) begin
		cycle_counter <= resetn ? cycle_counter + 1 : 0;
		if (resetn && trap) begin

			repeat (10) @(posedge clk);

			$display("TRAP after %1d clock cycles", cycle_counter);
			if (tests_passed) begin
				$display("ALL TESTS PASSED.");
				$finish;
			end else begin
				$display("ERROR!");
				if ($test$plusargs("noerror"))
					$finish;
				$stop;
			end
		end
	end



    `define SRAM s_mem.sram
	localparam DP = 2**16;
    integer  i, byte;
    
	reg [1023:0] firmware_file;
    reg [7:0] mem [0:200000];
	initial begin
		if (!$value$plusargs("firmware=%s", firmware_file))
			firmware_file = "firmware/firmware.hex";
		$readmemh(firmware_file, mem);


        for ( i = 0; i < DP; i = i + 1 ) begin
            for ( byte = 0; byte < 4; byte = byte + 1 ) begin
                if ( | mem[i*4+byte] ) begin
                    `SRAM[i][8*byte +: 8] = mem[i*4+byte];
                end
                else begin
                    `SRAM[i][8*byte +: 8] = 8'h0;
                end
            end
        end
	end


endmodule


