



`timescale 1 ns / 1 ps




module SimTop (

	input clock,
	input reset,

	output trap,
	input [31:0] irq,
	output tests_passed
);



	wire    mem_valid;
	wire    mem_ready;
	wire  [31:0] mem_addr;
	wire  [31:0] mem_wdata;
	wire  [3:0]  mem_wstrb;
	wire  [31:0] mem_rdata;








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

        .io_trace_valid(),
        .io_trace_bits()
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

	// picorv32_rvfimon rvfi_monitor (
	// 	.clock          (clk           ),
	// 	.reset          (!resetn       ),
	// 	.rvfi_valid     (rvfi_valid    ),
	// 	.rvfi_order     (rvfi_order    ),
	// 	.rvfi_insn      (rvfi_insn     ),
	// 	.rvfi_trap      (rvfi_trap     ),
	// 	.rvfi_halt      (rvfi_halt     ),
	// 	.rvfi_intr      (rvfi_intr     ),
	// 	.rvfi_rs1_addr  (rvfi_rs1_addr ),
	// 	.rvfi_rs2_addr  (rvfi_rs2_addr ),
	// 	.rvfi_rs1_rdata (rvfi_rs1_rdata),
	// 	.rvfi_rs2_rdata (rvfi_rs2_rdata),
	// 	.rvfi_rd_addr   (rvfi_rd_addr  ),
	// 	.rvfi_rd_wdata  (rvfi_rd_wdata ),
	// 	.rvfi_pc_rdata  (rvfi_pc_rdata ),
	// 	.rvfi_pc_wdata  (rvfi_pc_wdata ),
	// 	.rvfi_mem_addr  (rvfi_mem_addr ),
	// 	.rvfi_mem_rmask (rvfi_mem_rmask),
	// 	.rvfi_mem_wmask (rvfi_mem_wmask),
	// 	.rvfi_mem_rdata (rvfi_mem_rdata),
	// 	.rvfi_mem_wdata (rvfi_mem_wdata)
	// );




    
	reg [1023:0] firmware_file;
	initial begin
		if ($value$plusargs("%s", firmware_file)) begin
			$display("%s", firmware_file);
			$readmemh(firmware_file, s_mem.sram);
		end
	end





endmodule






