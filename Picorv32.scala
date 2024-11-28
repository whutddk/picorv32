/*
 *  PicoRV32 -- A Small RISC-V (RV32I) Processor Core
 *
 *  Copyright (C) 2015  Claire Xenia Wolf <claire@yosyshq.com>
 *
 *  Permission to use, copy, modify, and/or distribute this software for any
 *  purpose with or without fee is hereby granted, provided that the above
 *  copyright notice and this permission notice appear in all copies.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 *  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 *  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 *  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 *  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 *  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 */

/* verilator lint_off WIDTH */
/* verilator lint_off PINMISSING */
/* verilator lint_off CASEOVERLAP */
/* verilator lint_off CASEINCOMPLETE */

`timescale 1 ns / 1 ps
// `default_nettype none
// `define DEBUGNETS
// `define DEBUGREGS
// `define DEBUGASM
// `define DEBUG

`ifdef DEBUG
  `define debug(debug_command) debug_command
`else
  `define debug(debug_command)
`endif

`ifdef FORMAL
  `define FORMAL_KEEP (* keep *)
  `define assert(assert_expr) assert(assert_expr)
`else
  `ifdef DEBUGNETS
    `define FORMAL_KEEP (* keep *)
  `else
    `define FORMAL_KEEP
  `endif
  `define assert(assert_expr) empty_statement
`endif

// uncomment this for register file in extra module
// `define PICORV32_REGS picorv32_regs

// this macro can be used to check if the verilog files in your
// design are read in the correct order.
`define PICORV32_V


/***************************************************************
 * picorv32
 ***************************************************************/

package Picorv32

import chisel3._
import chisel3.util._

class Memory_Access_Bundle extends Bundle{
  val valid = Output(Bool())
	val ready = Input(Bool())

	val isInstr = Output(Bool())
	val addr  = Output(UInt(32.W))
	val wdata = Output(UInt(32.W))
	val wstrb = Output(UInt(3.W))
	val rdata = Input(UInt(32.W))

}

class Memory_LookAHead_Bundle extends Bundle{
	val read  = Bool()
	val write = Bool()
	val addr  = UInt(32.W)
	val wdata = UInt(32.W)
	val wstrb = UInt(4.W)
}

class PCPI_Access_Bundle extends Bundle{
	// Pico Co-Processor Interface (PCPI)
	val valid = Output(Bool())
	val ready = Input(Bool())
	
	val insn  = Output(UInt(32.W))
	val rs1   = Output(UInt(32.W))
	val rs2   = Output(UInt(32.W))
	val wr    = Input(Bool())
	val rd    = Input((UInt(32.W)))
	val wait  = Input(Bool())
}


class Riscv_Format_Bundle extends Bundle{
	valid = Bool()
	order = UInt(64.W)
	insn  = UInt(32.W)
	trap  = Bool()
	halt  = Bool()
	intr  = Bool()
	mode  = UInt(2.W)
	ixl   = UInt(2.W)
	rs1_addr  = UInt(5.W)
	rs2_addr  = UInt(5.W)
	rs1_rdata = UInt(32.W)
	rs2_rdata = UInt(32.W)
	rd_addr   = UInt(5.W)
	rd_wdata  = UInt(32.W)
	pc_rdata  = UInt(32.W)
	pc_wdata  = UInt(32.W)
	mem_addr  = UInt(32.W)
	mem_rmask = UInt(4.W)
	mem_wmask = UInt(4.W)
	mem_rdata = UInt(32.W)
	mem_wdata = UInt(32.W)

	csr_mcycle_rmask = UInt(64.W)
	csr_mcycle_wmask = UInt(64.W)
	csr_mcycle_rdata = UInt(64.W)
	csr_mcycle_wdata = UInt(64.W)

	csr_minstret_rmask= UInt(64.W)
	csr_minstret_wmask= UInt(64.W)
	csr_minstret_rdata= UInt(64.W)
	csr_minstret_wdata= UInt(64.W)
}

class Picorv32(
	ENABLE_COUNTERS: Boolean = true,
	ENABLE_COUNTERS64: Boolean = true,
	ENABLE_REGS_16_31: Boolean = true,
	ENABLE_REGS_DUALPORT: Boolean = true,
	LATCHED_MEM_RDATA: Boolean = false,
	TWO_STAGE_SHIFT: Boolean = true,
	BARREL_SHIFTER: Boolean = false,
	TWO_CYCLE_COMPARE: Boolean = false,
	TWO_CYCLE_ALU: Boolean = false,
	COMPRESSED_ISA: Boolean = false,
	CATCH_MISALIGN: Boolean = true,
	CATCH_ILLINSN: Boolean = true,
	ENABLE_PCPI: Boolean = false,
	ENABLE_MUL: Boolean = false,
	ENABLE_FAST_MUL: Boolean = false,
	ENABLE_DIV: Boolean = false,
	ENABLE_IRQ: Boolean = false,
	ENABLE_IRQ_QREGS: Boolean = true,
	ENABLE_IRQ_TIMER: Boolean = true,
	ENABLE_TRACE: Boolean = false,
	REGS_INIT_ZERO: Boolean = false,
	MASKED_IRQ: UInt = "h00000000".U(32.W),
	LATCHED_IRQ: UInt = "hffffffff".U(32.W),
	PROGADDR_RESET: UInt = "h00000000".U(32.W),
	PROGADDR_IRQ: UInt = "h00000010".U(32.W),
	STACKADDR: UInt = "hffffffff".U(32.W),
)
extends Module{

  class Picorv32IO extends Bundle{
	val trap = Output(Bool())
	val mem  = new Memory_Access_Bundle
	val mem_la = Output(new Memory_LookAHead_Bundle)
	val pcpi = new PCPI_Access_Bundle 

	val irq = Input(UInt(32.W))
	val eoi = Output(UInt(32.W))

	val rvfi = if( RISCV_FORMAL ){
		Some(Output(new Riscv_Format_Bundle))
	} else {
		None
	}

	val trace = Valid(UInt(36.W))
  }

  val io: Picorv32IO = IO(new Picorv32IO)

	val mem_valid = RegInit(false.B); io.mem.valid     := mem_valid
	val mem_instr = Reg(Bool()); io.mem.isInstr   := mem_instr
	val mem_addr  = Reg(UInt(32.W)); io.mem.addr  := mem_addr
	val mem_wdata = Reg(UInt(32.W)); io.mem.wdata := mem_wdata
	val mem_wstrb = Reg(UInt(4.W));  io.mem.wstrb := mem_wstrb



	val pcpi_valid = RegInit(false.B); io.pcpi.valid := pcpi_valid
	val pcpi_insn = Reg(UInt(32.W)); io.pcpi.insn := pcpi_insn

	val eoi = RegInit(0.U(32.W)); io.eoi := eoi

	val rvfi = Reg(new new Riscv_Format_Bundle); io.rvfi := rvfi

	val trace_valid = Reg(Bool()); io.trace.valid := trace_valid
	val trace_data = Reg(UInt(36.W)); io.trace.bits := trace_data



	def irq_timer    = 0
	def irq_ebreak   = 1
	def irq_buserror = 2

	def irqregs_offset = if(ENABLE_REGS_16_31) {32} else {16}
	def regfile_size   = if(ENABLE_REGS_16_31) {32} else { if(ENABLE_IRQ*ENABLE_IRQ_QREGS) {20} else {16}  }
	def regindex_bits  = if(ENABLE_REGS_16_31) {5 } else { if(ENABLE_IRQ*ENABLE_IRQ_QREGS) { 5} else { 4}  }

	def WITH_PCPI = ENABLE_PCPI || ENABLE_MUL || ENABLE_FAST_MUL || ENABLE_DIV;

	def TRACE_BRANCH = Cat( "b0001".U(4.W), 0.U(32.W) )
	def TRACE_ADDR   = Cat( "b0010".U(4.W), 0.U(32.W) )
	def TRACE_IRQ    = Cat( "b1000".U(4.W), 0.U(32.W) )


  val count_cycle = RegInit(0.U( (if(ENABLE_COUNTERS64){64} else{32}).W) )      
  if (ENABLE_COUNTERS){
    count_cycle := count_cycle + 1.U
  }

  val count_instr = RegInit(0.U(64.W))

  val reg_pc      = RegInit(PROGADDR_RESET(32.W))
  val reg_next_pc = RegInit(PROGADDR_RESET(32.W))
  val reg_op1 = Reg(UInt(32.W))
  val reg_op2 = Reg(UInt(32.W))
  val reg_out = if( ~STACKADDR ) { RegInit(STACKADDR(32.W)) } else { Reg(UInt(32.W)) }
  val reg_sh = Reg(UInt(5.W))



	val dbg_mem_valid = mem_valid
	val dbg_mem_instr = mem_instr
	val dbg_mem_ready = io.mem.ready

	val dbg_mem_addr  = mem_addr
	val dbg_mem_wdata = mem_wdata
	val dbg_mem_wstrb = mem_wstrb
	val dbg_mem_rdata = io.mem.rdata

	io.pcpi_rs1 := reg_op1
	io.pcpi_rs2 := reg_op2

	val next_pc = Wire(UInt(32.W))

	val irq_delay = RegInit(false.B)
	val irq_active = RegInit(false.B)
	val irq_mask = RegInit("hFFFFFFFF".U(32.W))
	val irq_pending = Reg(UInt(32.W))
	val timer = RegInit(0.U(32.W))


	// Internal PCPI Cores
	val pcpi_mul_wr    = Wire(Bool())
	val pcpi_mul_rd    = Wire(UInt(32.W))
	val pcpi_mul_wait  = Wire(Bool())
	val pcpi_mul_ready = Wire(Bool())

	val pcpi_div_wr    = Wire(Bool())
	val pcpi_div_rd    = Wire(UInt(32.W))
	val pcpi_div_wait  = Wire(Bool())
	val pcpi_div_ready = Wire(Bool())



	if(ENABLE_FAST_MUL){
		val pcpi_fast_mul = Module(new Pcpi_fast_mul)

		pcpi_fast_mul.io.valid := pcpi_valid
		pcpi_fast_mul.io.insn  := pcpi_insn
		pcpi_fast_mul.io.rs1   := pcpi_rs1
		pcpi_fast_mul.io.rs2   := pcpi_rs2

		pcpi_mul_wr    := pcpi_fast_mul.io.wr
		pcpi_mul_rd    := pcpi_fast_mul.io.rd
		pcpi_mul_wait  := pcpi_fast_mul.io.wait
		pcpi_mul_ready := pcpi_fast_mul.io.ready

	} else if(ENABLE_MUL){
		require(false, "Require Failed! Didnot implement")
		pcpi_mul_wr    := false.B
		pcpi_mul_rd    := 0.U
		pcpi_mul_wait  := false.B
		pcpi_mul_ready := false.B
	} else{
		pcpi_mul_wr    := false.B
		pcpi_mul_rd    := 0.U
		pcpi_mul_wait  := false.B
		pcpi_mul_ready := false.B
	}




	if (ENABLE_DIV) {
		val pcpi_div = Module(new Pcpi_div)

		pcpi_div.io.valid := pcpi_valid
		pcpi_div.io.insn  := pcpi_insn
		pcpi_div.io.rs1   := pcpi_rs1
		pcpi_div.io.rs2   := pcpi_rs2
		pcpi_div_wr     := pcpi_div.io.wr
		pcpi_div_rd     := pcpi_div.io.rd
		pcpi_div_wait   := pcpi_div.io.wait
		pcpi_div_ready  := pcpi_div.io.ready
	
	} else {
		pcpi_div_wr    := false.B
		pcpi_div_rd    := 0.U
		pcpi_div_wait  := false.B
		pcpi_div_ready := false.B
	}


	val pcpi_int_wr =
    Mux1H(Seq() ++
      if(ENABLE_PCPI) { Seq(pcpi_ready -> pcpi_wr) } ++
      if(ENABLE_MUL | ENABLE_FAST_MUL) { Seq(pcpi_mul_ready -> pcpi_mul_wr) } ++
      if(ENABLE_DIV) {Seq(pcpi_div_ready -> pcpi_div_wr)}
    )

	val pcpi_int_rd =
    Mux1H(Seq() ++ 
      if(ENABLE_PCPI) { Seq(pcpi_ready -> pcpi_rd) } ++
      if(ENABLE_MUL | ENABLE_FAST_MUL) { Seq(pcpi_mul_ready -> pcpi_mul_rd) } ++
      if(ENABLE_DIV) {Seq(pcpi_div_ready -> pcpi_div_rd)}
    )

	val pcpi_int_wait  = false.B | (if(ENABLE_PCPI)  {pcpi_wait }) | (if(ENABLE_MUL | ENABLE_FAST_MUL) {pcpi_mul_wait }) | (if(ENABLE_DIV) {pcpi_div_wait })
	val pcpi_int_ready = false.B | (if(ENABLE_PCPI)  {pcpi_ready}) | (if(ENABLE_MUL | ENABLE_FAST_MUL) {pcpi_mul_ready}) | (if(ENABLE_DIV) {pcpi_div_ready})

















	// Memory Interface
  
	val mem_wordsize = Reg(UInt(2.W))


	val mem_la_wdata = 
    Mux1H(Seq(
      ( mem_wordsize === 0.U ) -> reg_op2,
      ( mem_wordsize === 1.U ) -> Fill(2, reg_op2(15,0)),
      ( mem_wordsize === 2.U ) -> Fill(4, reg_op2( 7,0)),
    ))

	val mem_la_wstrb = 
    Mux1H(Seq(
      ( mem_wordsize === 0.U ) -> "b1111".U,
      ( mem_wordsize === 1.U ) -> Mux( reg_op1.extract(1), "b1100".U, "b0011".U),
      ( mem_wordsize === 2.U ) -> 1.U << reg_op1(1,0),
    ))

	val mem_rdata_word = 
    Mux1H(Seq(
      ( mem_wordsize === 0.U ) -> io.mem.rdata,
      ( mem_wordsize === 1.U ) -> Mux( reg_op1.extract(1), io.mem.rdata(31,16), io.mem.rdata(15, 0) )
      ( mem_wordsize === 2.U ) ->
        Mux1H(Seq(
          (reg_op1(1,0) === "b00".U) -> io.mem.rdata( 7, 0),
          (reg_op1(1,0) === "b01".U) -> io.mem.rdata(15, 8),
          (reg_op1(1,0) === "b10".U) -> io.mem.rdata(23,16),
          (reg_op1(1,0) === "b11".U) -> io.mem.rdata(31,24),
        ))
    ))

  io.mem_la.wdata := mem_la_wdata
  io.mem_la.wstrb := mem_la_wstrb





	val mem_state = RegInit(0.U(2.W))



	reg mem_do_prefetch;
	reg mem_do_rinst;
	reg mem_do_rdata;
	reg mem_do_wdata;

	val mem_xfer = Wire(Bool())

	val mem_la_secondword = RegInit(false.B)

	val mem_la_firstword =
    if(COMPRESSED_ISA) {(mem_do_prefetch | mem_do_rinst) & next_pc.extract(1) & ~mem_la_secondword} else {false.B}

	val mem_la_firstword_xfer =
    if(COMPRESSED_ISA) {mem_xfer & Mux(~last_mem_valid, mem_la_firstword, mem_la_firstword_reg)} else {false.B}

  val last_mem_valid = RegNext(mem_valid & ~io.mem.ready, false.B)
  val mem_la_firstword_reg = RegEnable(mem_la_firstword, false.B, ~last_mem_valid)






	val prefetched_high_word = RegInit(false.B)
	val clear_prefetched_high_word = Wire(Bool())
	reg [15:0] mem_16bit_buffer;
	reg [31:0] mem_rdata_q;

	val mem_rdata_latched = Wire(UInt(32.W))

	val mem_la_use_prefetched_high_word =
    if(COMPRESSED_ISA) {mem_la_firstword & prefetched_high_word & ~clear_prefetched_high_word} else {false.B}

	mem_xfer := (mem_valid & io.mem.ready) | (mem_la_use_prefetched_high_word & mem_do_rinst)

	val mem_busy = mem_do_prefetch | mem_do_rinst | mem_do_rdata | mem_do_wdata

	val mem_done = ~reset & ((mem_xfer & mem_state.orR & (mem_do_rinst | mem_do_rdata | mem_do_wdata)) | (mem_state.andR & mem_do_rinst)) &
			(~mem_la_firstword | (~mem_rdata_latched(1,0).andR & mem_xfer))

	io.mem_la.write := ~reset & (mem_state === 0.U) & mem_do_wdata
	io.mem_la.read  := ~reset & ((~mem_la_use_prefetched_high_word & (mem_state === 0.U) & (mem_do_rinst | mem_do_prefetch | mem_do_rdata)) |
			(if(COMPRESSED_ISA) {mem_xfer & Mux(~last_mem_valid, mem_la_firstword, mem_la_firstword_reg) & ~mem_la_secondword & mem_rdata_latched(1,0).andR} else {false.B}))

	io.mem_la.addr := Mux( mem_do_prefetch | mem_do_rinst, Cat(next_pc(31,2) + mem_la_firstword_xfer, 0.U(2.W)), Cat(reg_op1(31,2), 0.U(2.W)))

	val mem_rdata_latched_noshuffle =
    if( LATCHED_MEM_RDATA ){ io.mem.rdata } else { Mux(mem_xfer, io.mem.rdata, mem_rdata_q) }
    
	mem_rdata_latched :=
    if( COMPRESSED_ISA ){
      Mux(mem_la_use_prefetched_high_word, mem_16bit_buffer,
			  Mux(mem_la_secondword, Cat(mem_rdata_latched_noshuffle(15,0), mem_16bit_buffer),
			    Mux(mem_la_firstword, mem_rdata_latched_noshuffle(31,16), mem_rdata_latched_noshuffle)))
    } else {
      mem_rdata_latched_noshuffle
    }




	val next_insn_opcode = RegEnable( (if(COMPRESSED_ISA) {mem_rdata_latched} else {io.mem.rdata}), mem_xfer)



  printf("Warning\n")
  when(mem_xfer){
    mem_rdata_q := if(COMPRESSED_ISA) { mem_rdata_latched } else{ io.mem.rdata }
  }

  if (COMPRESSED_ISA){
    when(mem_done && (mem_do_prefetch | mem_do_rinst)){
      when(mem_rdata_latched(1,0) === "b00".U){
        when(mem_rdata_latched(15,13) === "b000".U){// C.ADDI4SPN
          mem_rdata_q(14,12) := "b000".U
          mem_rdata_q(31,20) := Cat(0.U(2.W), mem_rdata_latched(10,7), mem_rdata_latched(12,11), mem_rdata_latched.extract(5), mem_rdata_latched.extract(6), 0.U(2.W))
        } .elsewhen( mem_rdata_latched(15,13) === "b010".U ){// C.LW
          mem_rdata_q(31,20) := Cat(0.U(5.W), mem_rdata_latched.extract(5), mem_rdata_latched(12,10), mem_rdata_latched.extract(6), 0.U(2.W))
          mem_rdata_q(14,12) := "b010".U
        } .elsewhen( mem_rdata_latched(15,13) === "b110".U ){// C.SW
          mem_rdata_q(31,25) := Cat(0.U(5.W), mem_rdata_latched.extract(5), mem_rdata_latched.extract(12))
          mem_rdata_q(11: 7) := Cat(mem_rdata_latched(11,10), mem_rdata_latched.extract(6), 0.U(2.W)) 
          mem_rdata_q(14:12) := "b010".U
        }        
      } .elsewhen( mem_rdata_latched(1,0) === "b01".U ){
        when(mem_rdata_latched(15,13) === "b000".U){ // C.ADDI
          mem_rdata_q(14,12) := "b000".U
          mem_rdata_q(31,20) := Cat(mem_rdata_latched.extract(12), mem_rdata_latched(6,2)).asSInt
        } .elsewhen(mem_rdata_latched(15,13) === "b010".U){ // C.LI
          mem_rdata_q(14,12) := "b000".U
          mem_rdata_q(31,20) := Cat(mem_rdata_latched.extract(12), mem_rdata_latched(6,2)).asSInt
        } .elsewhen(mem_rdata_latched(15,13) === "b011".U){
          when(mem_rdata_latched(11,7) === 2.U) { // C.ADDI16SP
            mem_rdata_q(14,12) := "b000".U
            mem_rdata_q(31,20) := Cat(mem_rdata_latched.extract(12), mem_rdata_latched(4,3), mem_rdata_latched.extract(5), mem_rdata_latched.extract(2), mem_rdata_latched.extract(6), 0.U(4.W)).asSInt
            } .otherwise{ // C.LUI
            mem_rdata_q(31,12) := Cat(mem_rdata_latched.extract(12), mem_rdata_latched(6,2)).asSInt
            }
        } .elsewhen(mem_rdata_latched(15,13) === "b100".U){
          when(mem_rdata_latched(11,10) === "b00".U) { // C.SRLI
            mem_rdata_q(31,25) := "b0000000".U
            mem_rdata_q(14,12) := "b101".U
          }
          when(mem_rdata_latched(11,10) === "b01".U) { // C.SRAI
            mem_rdata_q(31,25) := "b0100000".U
            mem_rdata_q(14,12) := "b101".U
          }
          when(mem_rdata_latched(11,10) === "b10".U) { // C.ANDI
            mem_rdata_q(14,12) := "b111".U
            mem_rdata_q(31,20) := Cat(mem_rdata_latched.extract(12), mem_rdata_latched(6,2)).asSInt
          }
          when(mem_rdata_latched(12,10) === "b011".U) { // C.SUB, C.XOR, C.OR, C.AND
            when(mem_rdata_latched(6,5) === "b00".U) { mem_rdata_q(14,12) := "b000".U}
            when(mem_rdata_latched(6,5) === "b01".U) { mem_rdata_q(14,12) := "b100".U}
            when(mem_rdata_latched(6,5) === "b10".U) { mem_rdata_q(14,12) := "b110".U}
            when(mem_rdata_latched(6,5) === "b11".U) { mem_rdata_q(14,12) := "b111".U}
            mem_rdata_q(31,25) := Mux(mem_rdata_latched(6,5) === "b00".U, "b0100000".U, "b0000000".U)
          }
        } .elsewhen(mem_rdata_latched(15,13) === "b110".U){// C.BEQZ
          mem_rdata_q(14,12) := "b000".U
          { mem_rdata_q[31], mem_rdata_q[7], mem_rdata_q[30:25], mem_rdata_q[11:8] } <=
              $signed({mem_rdata_latched[12], mem_rdata_latched[6:5], mem_rdata_latched[2],
                  mem_rdata_latched[11:10], mem_rdata_latched[4:3]});
        } .elsewhen(mem_rdata_latched(15,13) === "b111".U){// C.BNEZ
          mem_rdata_q(14,12) := "b001".U
          { mem_rdata_q[31], mem_rdata_q[7], mem_rdata_q[30:25], mem_rdata_q[11:8] } <=
              $signed({mem_rdata_latched[12], mem_rdata_latched[6:5], mem_rdata_latched[2],
                  mem_rdata_latched[11:10], mem_rdata_latched[4:3]});            
        }        
      } .elsewhen( mem_rdata_latched(1,0) === "b10".U ){
        when(mem_rdata_latched(15,13) === "b000".U){// C.SLLI
          mem_rdata_q(31,25) := "b0000000".U
          mem_rdata_q(14,12) := "b001".U
        } .elsewhen( mem_rdata_latched(15,13) === "b010".U ){// C.LWSP
          mem_rdata_q(31,20) := Cat(0.U(4.W), mem_rdata_latched(3,2), mem_rdata_latched.extract(12), mem_rdata_latched(6,4), 0.U(2.W))
          mem_rdata_q(14,12) := "b010".U
        } .elsewhen( mem_rdata_latched(15,13) === "b100".U ){
          when(mem_rdata_latched.extract(12) === 0.U & mem_rdata_latched(6,2) === 0.U){ // C.JR
            mem_rdata_q(14,12) := "b000".U
            mem_rdata_q(31,20) := 0.U
          }
          when(mem_rdata_latched.extract(12) === 0.U & mem_rdata_latched(6,2) =/= 0.U){ // C.MV
            mem_rdata_q(14,12) := "b000".U
            mem_rdata_q(31,25) := "b0000000".U
          }
          when(mem_rdata_latched.extract(12) =/= 0.U & mem_rdata_latched(11,7) =/= 0.U & mem_rdata_latched(6,2) === 0.U){ // C.JALR
            mem_rdata_q(14,12) := "b000".U
            mem_rdata_q(31,20) := 0.U
          }
          when(mem_rdata_latched.extract(12) =/= 0.U & mem_rdata_latched(6,2) =/= 0.U){ // C.ADD
            mem_rdata_q(14,12) := "b000".U
            mem_rdata_q(31,25) := "b0000000".U
          }
        } .elsewhen( mem_rdata_latched(15,13) === "b110".U ){ // C.SWSP
            mem_rdata_q(31,25) := Cat(0.U(4.W), mem_rdata_latched(8,7), mem_rdata_latched(12,11))
            mem_rdata_q(11, 7) := Cat(mem_rdata_latched(11,9), 0.U(2.W))
            mem_rdata_q(14,12) := "b010".U
        }
      }
    }
  }



  when(~reset & ~io.trap) {
    when(mem_do_prefetch | mem_do_rinst | mem_do_rdata){
      assert(~mem_do_wdata)
    }
    when(mem_do_prefetch | mem_do_rinst){
      assert(~mem_do_rdata)
    }
    when(mem_do_rdata){
      assert(~mem_do_prefetch & ~mem_do_rinst)
    }
    when(mem_do_wdata){
      assert(~(mem_do_prefetch | mem_do_rinst | mem_do_rdata))
    }
    when(mem_state === 2.U | mem_state === 3.U){
      assert(mem_valid | mem_do_prefetch)
    }
  }


  when(io.trap){
    when(io.mem.ready){
      mem_valid := false.B
    }

    mem_la_secondword := false.B
    prefetched_high_word := false.B      
  } .otherwise{
    when(mem_la_read | mem_la_write) {
      mem_addr  := mem_la_addr
      mem_wstrb := mem_la_wstrb & Fill(4, mem_la_write)
    }
    when(mem_la_write) {
      mem_wdata := mem_la_wdata
    }

    when( mem_state === 0.U ){
      when(mem_do_prefetch | mem_do_rinst | mem_do_rdata){
        mem_valid := ~mem_la_use_prefetched_high_word
        mem_instr := mem_do_prefetch | mem_do_rinst
        mem_wstrb := 0.U
        mem_state := 1.U
      }
      when(mem_do_wdata){
        mem_valid := true.B
        mem_instr := false.B
        mem_state := 2.U
      }      
    } .elsewhen( mem_state === 1.U ){
      assert(mem_wstrb === 0.U)
      assert(mem_do_prefetch | mem_do_rinst | mem_do_rdata)
      assert(mem_valid === ~mem_la_use_prefetched_high_word)
      assert(mem_instr === (mem_do_prefetch | mem_do_rinst))
      when(mem_xfer){
        if (COMPRESSED_ISA) {
          when(mem_la_read){
            mem_valid := true.B
            mem_la_secondword := true.B
            when(~mem_la_use_prefetched_high_word){
              mem_16bit_buffer := mem_rdata(31,16)
            }
          } .otherwise{
            mem_valid := false.B
            mem_la_secondword := false.B
            when( ~mem_do_rdata ){
              when(~mem_rdata(1,0).andR | mem_la_secondword){
                mem_16bit_buffer := mem_rdata(31,16)
                prefetched_high_word := true.B
              } .otherwise{
                prefetched_high_word := false.B
              }            
            }

            mem_state := Mux(mem_do_rinst | mem_do_rdata, 0.U, 3.U)
          } 
        } else {
          mem_valid := false.B
          mem_la_secondword := false.B
          mem_state := Mux(mem_do_rinst | mem_do_rdata, 0.U, 3.U)
        }       
      }        
    } .elsewhen( mem_state === 2.U ){
        assert(mem_wstrb =/= 0.U)
        assert(mem_do_wdata)
        when(mem_xfer){
          mem_valid := false.B
          mem_state := 0.U
        }
    } .elsewhen( mem_state === 3.U ){
      assert(mem_wstrb === 0.U)
      assert(mem_do_prefetch)
      when(mem_do_rinst){
        mem_state := 0.U
      }     
    }
  }

  when(clear_prefetched_high_word){
    prefetched_high_word := false.B
  }





































	// Instruction Decoder







	val instr_beq  = RegInit(false.B)
  val instr_bne  = RegInit(false.B)
  val instr_blt  = RegInit(false.B)
  val instr_bge  = RegInit(false.B)
  val instr_bltu = RegInit(false.B)
  val instr_bgeu = RegInit(false.B)


  val instr_addi  = RegInit(false.B)
  val instr_slti  = RegInit(false.B)
  val instr_sltiu = RegInit(false.B)
  val instr_xori  = RegInit(false.B)
  val instr_ori   = RegInit(false.B)
  val instr_andi  = RegInit(false.B)
   
  val instr_add   = RegInit(false.B)
  val instr_sub   = RegInit(false.B)
  val instr_sll   = RegInit(false.B)
  val instr_slt   = RegInit(false.B)
  val instr_sltu  = RegInit(false.B)
  val instr_xor   = RegInit(false.B)
  val instr_srl   = RegInit(false.B)
  val instr_sra   = RegInit(false.B)
  val instr_or    = RegInit(false.B)
  val instr_and   = RegInit(false.B)
  val instr_fence = RegInit(false.B)


  val instr_lui = Reg(Bool())
  val instr_auipc = Reg(Bool())
  val instr_jal = Reg(Bool())
  val instr_jalr = Reg(Bool())
	val instr_lb = Reg(Bool())
  val instr_lh = Reg(Bool())
  val instr_lw = Reg(Bool())
  val instr_lbu = Reg(Bool())
  val instr_lhu = Reg(Bool())
  val instr_sb = Reg(Bool())
  val instr_sh = Reg(Bool())
  val instr_sw = Reg(Bool())
  val instr_slli = Reg(Bool())
  val instr_srli = Reg(Bool())
  val instr_srai = Reg(Bool())
	val instr_rdcycle = Reg(Bool())
  val instr_rdcycleh = Reg(Bool())
  val instr_rdinstr = Reg(Bool())
  val instr_rdinstrh = Reg(Bool())
  val instr_ecall_ebreak = Reg(Bool())
	val instr_getq = Reg(Bool())
  val instr_setq = Reg(Bool())
  val instr_retirq = Reg(Bool())
  val instr_maskirq = Reg(Bool())
  val instr_waitirq = Reg(Bool())
  val instr_timer = Reg(Bool())



  val decoded_rd  = Reg(UInt(regindex_bits.W))
  val decoded_rs1 = Reg(UInt(regindex_bits.W))
	val decoded_rs2 = Reg(UInt(5.W))

  val decoded_imm = Reg(UInt(32.W))
  val decoded_imm_j = Reg(UInt(32.W))

	val decoder_trigger  = Reg(Bool())
	val decoder_trigger_q  = Reg(Bool())
	val decoder_pseudo_trigger  = Reg(Bool())
	val decoder_pseudo_trigger_q  = Reg(Bool())
	val compressed_instr = Reg(Bool())

	val is_lui_auipc_jal = RegNext(instr_lui | instr_auipc | instr_jal)
	val is_lb_lh_lw_lbu_lhu = Reg(Bool())
	val is_slli_srli_srai = Reg(Bool())
	val is_jalr_addi_slti_sltiu_xori_ori_andi = Reg(Bool())
	val is_sb_sh_sw = Reg(Bool())
	val is_sll_srl_sra = Reg(Bool())
	val is_lui_auipc_jal_jalr_addi_add_sub = RegNext( Mux( decoder_trigger & ~decoder_pseudo_trigger, false.B, instr_lui | instr_auipc | instr_jal | instr_jalr | instr_addi | instr_add | instr_sub ) )
	val is_slti_blt_slt = RegNext(instr_slti | instr_blt | instr_slt)
	val is_sltiu_bltu_sltu = RegNext(instr_sltiu | instr_bltu | instr_sltu)


	val is_beq_bne_blt_bge_bltu_bgeu = RegInit(false.B)
	val is_lbu_lhu_lw = RegNext(instr_lbu | instr_lhu | instr_lw)


	val is_alu_reg_imm = Reg(Bool())
	val is_alu_reg_reg = Reg(Bool())
	val is_compare = RegNext( Mux( decoder_trigger & ~decoder_pseudo_trigger, false.B, is_beq_bne_blt_bge_bltu_bgeu | instr_slti | instr_slt | instr_sltiu | instr_sltu) , false.B)

	val instr_trap =
    if( CATCH_ILLINSN || WITH_PCPI ){
      ~(
        instr_lui | instr_auipc | instr_jal | instr_jalr |
        instr_beq | instr_bne | instr_blt | instr_bge | instr_bltu | instr_bgeu |
        instr_lb  | instr_lh | instr_lw | instr_lbu | instr_lhu | instr_sb | instr_sh | instr_sw |
        instr_addi| instr_slti | instr_sltiu | instr_xori | instr_ori | instr_andi | instr_slli | instr_srli | instr_srai |
        instr_add | instr_sub | instr_sll | instr_slt | instr_sltu | instr_xor | instr_srl | instr_sra | instr_or | instr_and |
        instr_rdcycle | instr_rdcycleh | instr_rdinstr | instr_rdinstrh | instr_fence |
        instr_getq| instr_setq | instr_retirq | instr_maskirq | instr_waitirq | instr_timer
      )
    } else { false.B }
 

	val is_rdcycle_rdcycleh_rdinstr_rdinstrh = instr_rdcycle | instr_rdcycleh | instr_rdinstr | instr_rdinstrh

	val dbg_insn_opcode = Wire(UInt(32.W))

	val dbg_ascii_instr = Wire(UInt(64.W));   dontTouch(dbg_ascii_instr) 
	val dbg_insn_imm    = Wire( UInt(32.W) ); dontTouch(dbg_insn_imm) 
	val dbg_insn_rs1    = Wire( UInt(5.W) );  dontTouch(dbg_insn_rs1) 
	val dbg_insn_rs2    = Wire( UInt(5.W) );  dontTouch(dbg_insn_rs2) 
	val dbg_insn_rd     = Wire( UInt(5.W) );  dontTouch(dbg_insn_rd) 

	dontTouch() reg [31:0] dbg_rs1val;
	dontTouch() reg [31:0] dbg_rs2val;
	dontTouch() reg dbg_rs1val_valid;
	dontTouch() reg dbg_rs2val_valid;



	val q_ascii_instr = RegNext( dbg_ascii_instr )
	val q_insn_imm    = RegNext( dbg_insn_imm )
	val q_insn_opcode = RegNext( dbg_insn_opcode )
	val q_insn_rs1    = RegNext( dbg_insn_rs1 )
	val q_insn_rs2    = RegNext( dbg_insn_rs2 )
	val q_insn_rd     = RegNext( dbg_insn_rd )
	val dbg_next      = RegNext( launch_next_insn )

	val launch_next_insn = Wire(Bool())
	val dbg_valid_insn = RegInit(false.B)

	when(io.trap){
		dbg_valid_insn := false.B
  } .elsewhen(launch_next_insn){
		dbg_valid_insn := true.B
  }


	val cached_ascii_instr = RegEnable(new_ascii_instr, decoder_trigger_q)
	val cached_insn_imm    = RegEnable(decoded_imm, decoder_trigger_q)
	val cached_insn_opcode = RegEnable( Mux( next_insn_opcode(1,0) === "b11".U, next_insn_opcode, next_insn_opcode(15,0) ), decoder_trigger_q)
	val cached_insn_rs1 = RegEnable(decoded_rs1, decoder_trigger_q)
	val cached_insn_rs2 = RegEnable(decoded_rs2, decoder_trigger_q)
	val cached_insn_rd  = RegEnable(decoded_rd,  decoder_trigger_q)

	val dbg_insn_addr  = RegEnable(next_pc, launch_next_insn)

  dbg_ascii_instr := Mux( dbg_next, Mux( decoder_pseudo_trigger_q, cached_ascii_instr, new_ascii_instr) ,q_ascii_instr  )
  dbg_insn_imm    := Mux( dbg_next, Mux( decoder_pseudo_trigger_q, cached_insn_imm, decoded_imm), q_insn_imm )
  dbg_insn_opcode := Mux( dbg_next, Mux( decoder_pseudo_trigger_q, cached_insn_opcode, Mux(&next_insn_opcode[1:0], next_insn_opcode, {16'b0, next_insn_opcode[15:0]}) ), q_insn_opcode)
  dbg_insn_rs1    := Mux( dbg_next, Mux( decoder_pseudo_trigger_q, cached_insn_rs1, decoded_rs1 ), q_insn_rs1 )
  dbg_insn_rs2    := Mux( dbg_next, Mux( decoder_pseudo_trigger_q, cached_insn_rs2, decoded_rs2 ), q_insn_rs2 )
  dbg_insn_rd     := Mux( dbg_next, Mux( decoder_pseudo_trigger_q, cached_insn_rd , decoded_rd  ), q_insn_rd  )


`ifdef DEBUGASM
		when(dbg_next){
			printf( s"debugasm %x %x %s", dbg_insn_addr, dbg_insn_opcode, Mux(dbg_ascii_instr, dbg_ascii_instr, "*") )
    }
`endif

`ifdef DEBUG
  when(dbg_next){
    when(&dbg_insn_opcode[1:0]){
      printf( s"DECODE: 0x%08x 0x%08x %-0s", dbg_insn_addr, dbg_insn_opcode, Mux(dbg_ascii_instr, dbg_ascii_instr, "UNKNOWN") )
    } .otherwise{
      printf( s"DECODE: 0x%08x 0x%04x %-0s", dbg_insn_addr, dbg_insn_opcode(15,0), Mux(dbg_ascii_instr, dbg_ascii_instr, "UNKNOWN") )
    }

  }
`endif

  when(mem_do_rinst && mem_done){
    instr_lui     := mem_rdata_latched(6,0) === "b0110111".U
    instr_auipc   := mem_rdata_latched(6,0) === "b0010111".U
    instr_jal     := mem_rdata_latched(6,0) === "b1101111".U
    instr_jalr    := mem_rdata_latched(6,0) === "b1100111".U & mem_rdata_latched(14,12) === "b000".U
    instr_retirq  := if(ENABLE_IRQ) {mem_rdata_latched(6,0) === "b0001011".U & mem_rdata_latched(31,25) === "b0000010".U} else {false.B}
    instr_waitirq := if(ENABLE_IRQ) {mem_rdata_latched(6,0) === "b0001011".U & mem_rdata_latched(31,25) === "b0000100".U} else {false.B}

    is_beq_bne_blt_bge_bltu_bgeu := mem_rdata_latched(6,0) === "b1100011".U
    is_lb_lh_lw_lbu_lhu          := mem_rdata_latched(6,0) === "b0000011".U
    is_sb_sh_sw                  := mem_rdata_latched(6,0) === "b0100011".U
    is_alu_reg_imm               := mem_rdata_latched(6,0) === "b0010011".U
    is_alu_reg_reg               := mem_rdata_latched(6,0) === "b0110011".U

    {
      val temp = Cat( Fill(11, mem_rdata_latched.extract(31) ) ,mem_rdata_latched(31:12), 0.U(1.W) )
      decoded_imm_j := Cat( temp(31,20), temp(8,1), temp.extract(9), temp(19,10), temp.extract(0) )
    }


    decoded_rd  := mem_rdata_latched(11,7)
    decoded_rs1 := mem_rdata_latched(19,15)
    decoded_rs2 := mem_rdata_latched(24,20)

    if(ENABLE_IRQ && ENABLE_IRQ_QREGS){
      when(mem_rdata_latched(6,0) === "b0001011".U & mem_rdata_latched(31,25) === "b0000000".U ){
        decoded_rs1 := Cat( 1.U(1.W), mem_rdata_latched(19,15) ) // instr_getq
      }        
    }

    if(ENABLE_IRQ){
      when(mem_rdata_latched(6,0) === "b0001011".U & mem_rdata_latched(31,25) === "b0000010".U ){
        decoded_rs1 := if(ENABLE_IRQ_QREGS) {irqregs_offset} else {3.U}; // instr_retirq
      }

    }


    compressed_instr := 
      if (COMPRESSED_ISA){ mem_rdata_latched(1,0) =/= "b11".U } else { false.B }



    if (COMPRESSED_ISA){
      when(mem_rdata_latched(1,0) =/= "b11".U){
        decoded_rd  := 0.U
        decoded_rs1 := 0.U
        decoded_rs2 := 0.U

        { decoded_imm_j[31:11], decoded_imm_j[4], decoded_imm_j[9:8], decoded_imm_j[10], decoded_imm_j[6],
          decoded_imm_j[7], decoded_imm_j[3:1], decoded_imm_j[5], decoded_imm_j[0] } <= $signed({mem_rdata_latched[12:2], 1'b0});

        when( mem_rdata_latched(1,0) === "b00".U){ // Quadrant 0
          when(mem_rdata_latched(15,13) === "b000".U){ // C.ADDI4SPN
            is_alu_reg_imm := mem_rdata_latched(12,5).orR
            decoded_rs1 := 2.U
            decoded_rd := 8.U + mem_rdata_latched(4,2)
          } .elsewhen(mem_rdata_latched(15,13) === "b010".U){ // C.LW
            is_lb_lh_lw_lbu_lhu := 1.U
            decoded_rs1 := 8.U + mem_rdata_latched(9,7)
            decoded_rd  := 8.U + mem_rdata_latched(4,2)
          } .elsewhen(mem_rdata_latched(15,13) === "b110".U){ // C.SW
            is_sb_sh_sw := 1.U
            decoded_rs1 := 8.U + mem_rdata_latched(9,7)
            decoded_rs2 := 8.U + mem_rdata_latched(4,2)
          }           
        } .elsewhen( mem_rdata_latched(1,0) === "b01".U){ // Quadrant 1
          when(mem_rdata_latched(15,13) === "b000".U){ // C.NOP / C.ADDI
            is_alu_reg_imm := 1.U
            decoded_rd  := mem_rdata_latched(11,7)
            decoded_rs1 := mem_rdata_latched(11,7)
          } .elsewhen( mem_rdata_latched(15,13) === "b001".U ){ // C.JAL
            instr_jal  := 1.U
            decoded_rd := 1.U
          } .elsewhen( mem_rdata_latched(15,13) === "b010".U ){ // C.LI
            is_alu_reg_imm := 1.U
            decoded_rd  := mem_rdata_latched(11,7)
            decoded_rs1 := 0.U
          } .elsewhen( mem_rdata_latched(15,13) === "b011".U ){
            when(mem_rdata_latched.extract(12) | mem_rdata_latched(6,2) ){
              when(mem_rdata_latched[11:7] == 2){ // C.ADDI16SP
                is_alu_reg_imm := 1.U
                decoded_rd     := mem_rdata_latched(11,7)
                decoded_rs1    := mem_rdata_latched(11,7)
              } .otherwise{ // C.LUI
                instr_lui   := 1.U
                decoded_rd  := mem_rdata_latched(11,7)
                decoded_rs1 := 0.U
              }                
            }
          } .elsewhen( mem_rdata_latched(15,13) === "b100".U ){
            when( ~mem_rdata_latched.extract(11) & ~mem_rdata_latched.extract(12) ){ // C.SRLI, C.SRAI
              is_alu_reg_imm := 1.U
              decoded_rd  := 8.U + mem_rdata_latched(9,7)
              decoded_rs1 := 8.U + mem_rdata_latched(9,7)
              decoded_rs2 := Cat(mem_rdata_latched.extract(12), mem_rdata_latched(6,2))
            }
            when(mem_rdata_latched(11,10) === "b10".U){ // C.ANDI
              is_alu_reg_imm := 1.U
              decoded_rd     := 8.U + mem_rdata_latched(9,7)
              decoded_rs1    := 8.U + mem_rdata_latched(9,7)
            }
            when(mem_rdata_latched(12,10) === "b011".U){ // C.SUB, C.XOR, C.OR, C.AND
              is_alu_reg_reg := 1.U
              decoded_rd     := 8.U + mem_rdata_latched(9,7)
              decoded_rs1    := 8.U + mem_rdata_latched(9,7)
              decoded_rs2    := 8.U + mem_rdata_latched(4,2)
            }
          } .elsewhen( mem_rdata_latched(15,13) === "b101".U ){ // C.J
            instr_jal := 1.U
          } .elsewhen( mem_rdata_latched(15,13) === "b110".U ){ // C.BEQZ
            is_beq_bne_blt_bge_bltu_bgeu := 1.U
            decoded_rs1 := 8.U + mem_rdata_latched(9,7)
            decoded_rs2 := 0.U
          } .elsewhen( mem_rdata_latched(15,13) === "b111".U ){ // C.BNEZ
            is_beq_bne_blt_bge_bltu_bgeu := 1.U
            decoded_rs1 := 8.U + mem_rdata_latched(9,7)
            decoded_rs2 := 0.U
          }
        } .elsewhen( mem_rdata_latched(1,0) === "b10".U){ // Quadrant 2
          when(mem_rdata_latched(15,13) === "b000".U){ // C.SLLI
            when( ~mem_rdata_latched.extract(12) ){
              is_alu_reg_imm := 1.U
              decoded_rd  := mem_rdata_latched(11,7)
              decoded_rs1 := mem_rdata_latched(11,7)
              decoded_rs2 := Cat(mem_rdata_latched.extract(12), mem_rdata_latched(6,2))
            }            
          } .elsewhen(mem_rdata_latched(15,13) === "b010".U){ // C.LWSP
            when(mem_rdata_latched(11,7)){
              is_lb_lh_lw_lbu_lhu := 1.U
              decoded_rd  := mem_rdata_latched(11,7)
              decoded_rs1 := 2.U
            }             
          } .elsewhen(mem_rdata_latched(15,13) === "b100".U){
            when( mem_rdata_latched.extract(12) === 0.U & mem_rdata_latched(11,7) =/= 0.U & mem_rdata_latched(6,2) === 0.U){ // C.JR
              instr_jalr  := 1.U
              decoded_rd  := 0.U
              decoded_rs1 := mem_rdata_latched(11,7)
            }
            when(mem_rdata_latched.extract(12) === 0.U & mem_rdata_latched(6,2) =/= 0.U){ // C.MV
              is_alu_reg_reg := 1.U
              decoded_rd  := mem_rdata_latched(11,7)
              decoded_rs1 := 0.U
              decoded_rs2 := mem_rdata_latched(6,2)
            }
            when(mem_rdata_latched.extract(12) =/= 0.U & mem_rdata_latched(11,7) =/= 0.U & mem_rdata_latched(6,2) === 0.U){ // C.JALR
              instr_jalr  := 1.U
              decoded_rd  := 1.U
              decoded_rs1 := mem_rdata_latched(11,7)
            }
            when(mem_rdata_latched.extract(12) =/= 0.U & mem_rdata_latched(6,2) =/= 0){ // C.ADD
              is_alu_reg_reg := 1.U
              decoded_rd  := mem_rdata_latched(11,7)
              decoded_rs1 := mem_rdata_latched(11,7)
              decoded_rs2 := mem_rdata_latched(6,2)
            }        
          } .elsewhen(mem_rdata_latched(15,13) === "b110".U){// C.SWSP
              is_sb_sh_sw := 1.U
              decoded_rs1 := 2.U
              decoded_rs2 := mem_rdata_latched(6,2)
          }
        }         
      }
    }   
  }

  when(decoder_trigger && !decoder_pseudo_trigger){
    pcpi_insn := if(WITH_PCPI) {mem_rdata_q} else {0.U}

    instr_beq   := is_beq_bne_blt_bge_bltu_bgeu & mem_rdata_q(14:12) === "b000".U
    instr_bne   := is_beq_bne_blt_bge_bltu_bgeu & mem_rdata_q(14:12) === "b001".U
    instr_blt   := is_beq_bne_blt_bge_bltu_bgeu & mem_rdata_q(14:12) === "b100".U
    instr_bge   := is_beq_bne_blt_bge_bltu_bgeu & mem_rdata_q(14:12) === "b101".U
    instr_bltu  := is_beq_bne_blt_bge_bltu_bgeu & mem_rdata_q(14:12) === "b110".U
    instr_bgeu  := is_beq_bne_blt_bge_bltu_bgeu & mem_rdata_q(14:12) === "b111".U

    instr_lb    := is_lb_lh_lw_lbu_lhu & mem_rdata_q(14,12) === "b000".U
    instr_lh    := is_lb_lh_lw_lbu_lhu & mem_rdata_q(14,12) === "b001".U
    instr_lw    := is_lb_lh_lw_lbu_lhu & mem_rdata_q(14,12) === "b010".U
    instr_lbu   := is_lb_lh_lw_lbu_lhu & mem_rdata_q(14,12) === "b100".U
    instr_lhu   := is_lb_lh_lw_lbu_lhu & mem_rdata_q(14,12) === "b101".U

    instr_sb    := is_sb_sh_sw & mem_rdata_q(14,12) === "b000".U
    instr_sh    := is_sb_sh_sw & mem_rdata_q(14,12) === "b001".U
    instr_sw    := is_sb_sh_sw & mem_rdata_q(14,12) === "b010".U

    instr_addi  := is_alu_reg_imm & mem_rdata_q(14,12) === "b000".U
    instr_slti  := is_alu_reg_imm & mem_rdata_q(14,12) === "b010".U
    instr_sltiu := is_alu_reg_imm & mem_rdata_q(14,12) === "b011".U
    instr_xori  := is_alu_reg_imm & mem_rdata_q(14,12) === "b100".U
    instr_ori   := is_alu_reg_imm & mem_rdata_q(14,12) === "b110".U
    instr_andi  := is_alu_reg_imm & mem_rdata_q(14,12) === "b111".U

    instr_slli  := is_alu_reg_imm & mem_rdata_q(14,12) === "b001".U & mem_rdata_q(31,25) === "b0000000".U
    instr_srli  := is_alu_reg_imm & mem_rdata_q(14,12) === "b101".U & mem_rdata_q(31,25) === "b0000000".U
    instr_srai  := is_alu_reg_imm & mem_rdata_q(14,12) === "b101".U & mem_rdata_q(31,25) === "b0100000".U

    instr_add   := is_alu_reg_reg & mem_rdata_q(14,12) === "b000".U & mem_rdata_q(31,25) === "b0000000".U
    instr_sub   := is_alu_reg_reg & mem_rdata_q(14,12) === "b000".U & mem_rdata_q(31,25) === "b0100000".U
    instr_sll   := is_alu_reg_reg & mem_rdata_q(14,12) === "b001".U & mem_rdata_q(31,25) === "b0000000".U
    instr_slt   := is_alu_reg_reg & mem_rdata_q(14,12) === "b010".U & mem_rdata_q(31,25) === "b0000000".U
    instr_sltu  := is_alu_reg_reg & mem_rdata_q(14,12) === "b011".U & mem_rdata_q(31,25) === "b0000000".U
    instr_xor   := is_alu_reg_reg & mem_rdata_q(14,12) === "b100".U & mem_rdata_q(31,25) === "b0000000".U
    instr_srl   := is_alu_reg_reg & mem_rdata_q(14,12) === "b101".U & mem_rdata_q(31,25) === "b0000000".U
    instr_sra   := is_alu_reg_reg & mem_rdata_q(14,12) === "b101".U & mem_rdata_q(31,25) === "b0100000".U
    instr_or    := is_alu_reg_reg & mem_rdata_q(14,12) === "b110".U & mem_rdata_q(31,25) === "b0000000".U
    instr_and   := is_alu_reg_reg & mem_rdata_q(14,12) === "b111".U & mem_rdata_q(31,25) === "b0000000".U

    if( ENABLE_COUNTERS ){
      instr_rdcycle := 
        (mem_rdata_q(6,0) === "b1110011".U & mem_rdata_q(31,12) === "b11000000000000000010".U) |
        (mem_rdata_q(6,0) === "b1110011".U & mem_rdata_q(31,12) === "b11000000000100000010".U)
      instr_rdinstr := mem_rdata_q(6,0) === "b1110011".U & mem_rdata_q(31,12) === "b11000000001000000010".U
    } else {
      instr_rdcycle  := false.B
      instr_rdinstr  := false.B
    }

    if( ENABLE_COUNTERS & ENABLE_COUNTERS64 ){
      instr_rdcycleh :=
        (mem_rdata_q(6,0) === "b1110011".U & mem_rdata_q(31,12) === "b11001000000000000010".U) |
        (mem_rdata_q(6,0) === "b1110011".U & mem_rdata_q(31,12) === "b11001000000100000010".U)
      instr_rdinstrh := mem_rdata_q(6,0) === "b1110011".U & mem_rdata_q(31,12) === "b11001000001000000010".U
    } else {
      instr_rdcycleh := false.B
      instr_rdinstrh := false.B  
    }

    instr_ecall_ebreak := 
      (mem_rdata_q(6,0) === "b1110011".U & mem_rdata_q(31,21) === 0.U & mem_rdata_q(19,7) === 0.U) |
      ( if(COMPRESSED_ISA) {mem_rdata_q(15,0) === "h9002".U} else {false.B})

    instr_fence := mem_rdata_q(6,0) === "b0001111".U & mem_rdata_q(14,12) === 0.U

    if( ENABLE_IRQ ){
      instr_maskirq := mem_rdata_q(6,0) === "b0001011".U & mem_rdata_q(31,25) === "b0000011".U
    } else {
      instr_maskirq := false.B
    }

    if( ENABLE_IRQ & ENABLE_IRQ_QREGS ){
      instr_getq    := mem_rdata_q(6,0) === "b0001011".U & mem_rdata_q(31,25) === "b0000000".U
      instr_setq    := mem_rdata_q(6,0) === "b0001011".U & mem_rdata_q(31,25) === "b0000001".U
      instr_timer   := mem_rdata_q(6,0) === "b0001011".U & mem_rdata_q(31,25) === "b0000101".U
    } else {
      instr_getq    := false.B
      instr_setq    := false.B
      instr_timer   := false.B
    }


    is_slli_srli_srai := is_alu_reg_imm & (
      ( mem_rdata_q(14,12) === "b001".U & mem_rdata_q(31,25) === "b0000000".U ) |
      ( mem_rdata_q(14,12) === "b101".U & mem_rdata_q(31,25) === "b0000000".U ) |
      ( mem_rdata_q(14,12) === "b101".U & mem_rdata_q(31,25) === "b0100000".U )
    )


    is_jalr_addi_slti_sltiu_xori_ori_andi := instr_jalr | (
      is_alu_reg_imm & (
        ( mem_rdata_q(14,12) === "b000".U ) |
        ( mem_rdata_q(14,12) === "b010".U ) |
        ( mem_rdata_q(14,12) === "b011".U ) |
        ( mem_rdata_q(14,12) === "b100".U ) |
        ( mem_rdata_q(14,12) === "b110".U ) |
        ( mem_rdata_q(14,12) === "b111".U )        
      )
    )

    is_sll_srl_sra := is_alu_reg_reg & (
      ( mem_rdata_q(14,12) === "b001".U & mem_rdata_q(31,25) === "b0000000".U ) |
      ( mem_rdata_q(14,12) === "b101".U & mem_rdata_q(31,25) === "b0000000".U ) |
      ( mem_rdata_q(14,12) === "b101".U & mem_rdata_q(31,25) === "b0100000".U )        
    )

    decoded_imm := Mux1H(Seq(
      instr_jal                                           -> decoded_imm_j,
      (instr_lui | instr_auipc)                           -> mem_rdata_q(31,12) << 12,
      (instr_jalr | is_lb_lh_lw_lbu_lhu | is_alu_reg_imm) -> Cat( Fill( 20, mem_rdata_q.extract(31)), mem_rdata_q(31,20) ),
      is_beq_bne_blt_bge_bltu_bgeu                        -> Cat( Fill( 19, mem_rdata_q.extract(31)), mem_rdata_q.extract(31), mem_rdata_q.extract(7), mem_rdata_q[30:25], mem_rdata_q[11:8], 0.U(1.W)),
      is_sb_sh_sw                                         -> Cat( Fill( 20, mem_rdata_q.extract(31)), mem_rdata_q(31,25), mem_rdata_q(11,7)),
    ))
  }











































	// Main State Machine

	val cpu_state_trap   = "b10000000".U
	val cpu_state_fetch  = "b01000000".U
	val cpu_state_ld_rs1 = "b00100000".U
	val cpu_state_ld_rs2 = "b00010000".U
	val cpu_state_exec   = "b00001000".U
	val cpu_state_shift  = "b00000100".U
	val cpu_state_stmem  = "b00000010".U
	val cpu_state_ldmem  = "b00000001".U

	val cpu_state = RegInit(cpu_state_fetch(8.W));
	val irq_state = RegInit(0.U(2.W))



	reg set_mem_do_rinst;
	reg set_mem_do_rdata;
	reg set_mem_do_wdata;

	val latched_store  = RegInit((if(~STACKADDR){true.B}else{false.B}))
	val latched_stalu  = RegInit(false.B)
	val latched_branch = RegInit(false.B)
	reg latched_compr;
	val latched_trace = RegInit(false.B)
	val latched_is_lu = RegInit(false.B)
	val latched_is_lh = RegInit(false.B)
	val latched_is_lb = RegInit(false.B)

	val latched_rd = if(~STACKADDR){RegInit(2.U(regindex_bits.W))} else { Reg(UInt((regindex_bits.W))) }

	reg [31:0] current_pc;
	assign next_pc = latched_store && latched_branch ? reg_out & ~1 : reg_next_pc;

	reg [3:0] pcpi_timeout_counter;
	val pcpi_timeout = RegInit(false.B)

	val next_irq_pending = RegInit(0.U(32.W))
	reg do_waitirq;

	reg [31:0] alu_out_q;
	reg alu_out_0_q;
	reg alu_wait, alu_wait_2;

	val alu_add_sub = Wire(UInt(32.W))
	val alu_shl = Wire(UInt(32.W))
  val alu_shr = Wire(UInt(32.W))
	val alu_eq  = Wire(Bool())
  val alu_ltu = Wire(Bool())
  val alu_lts = Wire(Bool())





	if (TWO_CYCLE_ALU){
    alu_add_sub := RegNext( Mux(instr_sub, reg_op1 - reg_op2, reg_op1 + reg_op2) )
    alu_eq  := RegNext(reg_op1 === reg_op2)
    alu_lts := RegNext(reg_op1.asSInt < reg_op2.asSInt)
    alu_ltu := RegNext(reg_op1 < reg_op2)
    alu_shl := RegNext(reg_op1 << reg_op2(4,0))
    alu_shr := RegNext( Cat( Fill(32, Mux(instr_sra | instr_srai, reg_op1.extract(31), 0.U(1.W))), reg_op1) >> reg_op2(4,0) )
  } else{
    alu_add_sub := Mux( instr_sub, reg_op1 - reg_op2, reg_op1 + reg_op2 )
    alu_eq  := reg_op1 === reg_op2
    alu_lts := reg_op1.asSInt < reg_op2.asSInt
    alu_ltu := reg_op1 < reg_op2
    alu_shl := reg_op1 << reg_op2(4,0)
    alu_shr := Cat( Fill(32, Mux(instr_sra | instr_srai, reg_op1.extract(31), 0.U(1.W))), reg_op1) >> reg_op2(4,0)
  }

  val alu_out_0 = Mux1H(Seq(
      instr_beq  ->  alu_eq,
      instr_bne  -> ~alu_eq,
      instr_bge  -> ~alu_lts,
      instr_bgeu -> ~alu_ltu,
      ( is_slti_blt_slt    & ~instr_beq & ~instr_bne & ~instr_bge & ~instr_bgeu ) -> alu_lts,
      ( is_sltiu_bltu_sltu & ~instr_beq & ~instr_bne & ~instr_bge & ~instr_bgeu ) -> alu_ltu,
    ) ++
    if( !TWO_CYCLE_COMPARE ){
      Seq(
        is_slti_blt_slt     -> alu_lts,
        is_sltiu_bltu_sltu  -> alu_ltu,      
      )
    } else { Seq() }
  )

  val alu_out = Mux1H(Seq(
			is_lui_auipc_jal_jalr_addi_add_sub -> alu_add_sub,
			is_compare -> alu_out_0,
			( instr_xori | instr_xor ) -> ( reg_op1 ^ reg_op2 ),
			( instr_ori  | instr_or  ) -> ( reg_op1 | reg_op2 ),
			( instr_andi | instr_and ) -> ( reg_op1 & reg_op2 ),
    ) ++ if( BARREL_SHIFTER ){
			(instr_sll | instr_slli) -> alu_shl,
			(instr_srl | instr_srli | instr_sra | instr_srai) -> alu_shr,
    } else { Seq() }
  )




	val clear_prefetched_high_word_q = RegNext( clear_prefetched_high_word )

  clear_prefetched_high_word :=
    MuxCase( clear_prefetched_high_word_q, Array(
      (~prefetched_high_word -> false.B) ++ 
      if( COMPRESSED_ISA ){
        Array( latched_branch | irq_state | reset)
      } else { Array() }
    ) )







  val cpuregs_write = 
    (cpu_state === cpu_state_fetch) & (
      latched_branch | latched_store |
      (if( ENABLE_IRQ ) { irq_state.extract(0) | irq_state.extract(1) } else { false.B })
    )

	val cpuregs_wrdata =
    Mux1H(Seq(
        (cpu_state == cpu_state_fetch) & latched_branch                  -> reg_pc + Mux(latched_compr, 2.U, 4.U),
        (cpu_state == cpu_state_fetch) & latched_store & ~latched_branch -> Mux(latched_stalu, alu_out_q, reg_out),
      ) ++ if( ENABLE_IRQ ) { Seq(
        (cpu_state == cpu_state_fetch) & irq_state.extract(0) -> reg_next_pc | latched_compr,
        (cpu_state == cpu_state_fetch) & irq_state.extract(1) -> irq_pending & ~irq_mask        
      )} else { Seq() }
    )


	wire[31:0] cpuregs_rdata1;
	wire[31:0] cpuregs_rdata2;

  val cpuregs_waddr = latched_rd



	val cpuregs_raddr1 = if(ENABLE_REGS_DUALPORT) {decoded_rs1} else { decoded_rs }
	val cpuregs_raddr2 = if(ENABLE_REGS_DUALPORT) {decoded_rs2} else { 0.U(6.W) }

  val cpuregs = Module(new MEM(UInt(32.W), 32))

  when( ~reset & cpuregs_write & latched_rd ){
    cpuregs(cpuregs_waddr(4,0)) := cpuregs_wrdata
  }

  cpuregs_rdata1 := cpuregs(cpuregs_raddr1(4,0))
  cpuregs_rdata2 := cpuregs(cpuregs_raddr2(4,0))


	val cpuregs_rs1 = Wire(UInt(32.W))
	val cpuregs_rs2 = Wire(UInt(32.W))
	val decoded_rs  = Wire(UInt(regindex_bits.W))

  if (ENABLE_REGS_DUALPORT) {
    cpuregs_rs1 := Mux(decoded_rs1, cpuregs_rdata1, 0.U)
    cpuregs_rs2 := Mux(decoded_rs2, cpuregs_rdata2, 0.U)
  } else {
    decoded_rs  := Mux(cpu_state === cpu_state_ld_rs2, decoded_rs2, decoded_rs1)
    cpuregs_rs1 := Mux(decoded_rs, cpuregs_rdata1, 0.U)
    cpuregs_rs2 := cpuregs_rs1
  }

	launch_next_insn := 
    if( !ENABLE_IRQ ){
      (cpu_state === cpu_state_fetch) & decoder_trigger
    } else {
      (cpu_state === cpu_state_fetch) & decoder_trigger & ( irq_delay | irq_active | ~(irq_pending & ~irq_mask))
    }
  
























	always @(posedge clk) begin
		reg_sh <= 'bx;
		reg_out <= 'bx;
		set_mem_do_rinst = 0;
		set_mem_do_rdata = 0;
		set_mem_do_wdata = 0;

		alu_out_0_q <= alu_out_0;
		alu_out_q <= alu_out;

		alu_wait <= 0;
		alu_wait_2 <= 0;

		if (launch_next_insn) begin
			dbg_rs1val <= 'bx;
			dbg_rs2val <= 'bx;
			dbg_rs1val_valid <= 0;
			dbg_rs2val_valid <= 0;
		end

		if (WITH_PCPI && CATCH_ILLINSN){
			when(resetn && pcpi_valid && !pcpi_int_wait){
				if (pcpi_timeout_counter)
					pcpi_timeout_counter <= pcpi_timeout_counter - 1;        
      } .otherwise{
				pcpi_timeout_counter <= ~0;        
      }
			pcpi_timeout <= !pcpi_timeout_counter;      
    }




    if( ENABLE_IRQ ){
		  next_irq_pending := irq_pending & LATCHED_IRQ
    }

		if (ENABLE_IRQ && ENABLE_IRQ_TIMER){
      when( timer =/= 0.U ){
			  timer := timer - 1.U
      }
    }

		decoder_trigger <= mem_do_rinst && mem_done;
		decoder_trigger_q <= decoder_trigger;
		decoder_pseudo_trigger <= 0;
		decoder_pseudo_trigger_q <= decoder_pseudo_trigger;
		do_waitirq <= 0;

		trace_valid <= 0;

		if (!ENABLE_TRACE){
			trace_data <= 'bx;      
    }


    io.trap := RegNext(cpu_state === cpu_state_trap)


    when( cpu_state_fetch === cpu_state ){
      mem_do_rinst <= !decoder_trigger && !do_waitirq;
      mem_wordsize <= 0;

      current_pc = reg_next_pc;

      (* parallel_case *)
      case (1'b1)
        latched_branch: begin
          current_pc = latched_store ? (latched_stalu ? alu_out_q : reg_out) & ~1 : reg_next_pc;
          `debug($display("ST_RD:  %2d 0x%08x, BRANCH 0x%08x", latched_rd, reg_pc + (latched_compr ? 2 : 4), current_pc);)
        end
        latched_store && !latched_branch: begin
          `debug($display("ST_RD:  %2d 0x%08x", latched_rd, latched_stalu ? alu_out_q : reg_out);)
        end
        ENABLE_IRQ && irq_state[0]: begin
          current_pc = PROGADDR_IRQ;
          irq_active <= 1;
          mem_do_rinst <= 1;
        end
        ENABLE_IRQ && irq_state[1]: begin
          eoi <= irq_pending & ~irq_mask;
          next_irq_pending = next_irq_pending & irq_mask;
        end
      endcase

      if (ENABLE_TRACE && latched_trace) begin
        latched_trace <= 0;
        trace_valid <= 1;
        if (latched_branch)
          trace_data <= (irq_active ? TRACE_IRQ : 0) | TRACE_BRANCH | (current_pc & 32'hfffffffe);
        else
          trace_data <= (irq_active ? TRACE_IRQ : 0) | (latched_stalu ? alu_out_q : reg_out);
      end

      reg_pc <= current_pc;
      reg_next_pc <= current_pc;

      latched_store <= 0;
      latched_stalu <= 0;
      latched_branch <= 0;
      latched_is_lu <= 0;
      latched_is_lh <= 0;
      latched_is_lb <= 0;
      latched_rd <= decoded_rd;
      latched_compr <= compressed_instr;

      if (ENABLE_IRQ && ((decoder_trigger && !irq_active && !irq_delay && |(irq_pending & ~irq_mask)) || irq_state)) begin
        irq_state <=
          irq_state == 2'b00 ? 2'b01 :
          irq_state == 2'b01 ? 2'b10 : 2'b00;
        latched_compr <= latched_compr;
        if (ENABLE_IRQ_QREGS)
          latched_rd <= irqregs_offset | irq_state[0];
        else
          latched_rd <= irq_state[0] ? 4 : 3;
      end else
      if (ENABLE_IRQ && (decoder_trigger || do_waitirq) && instr_waitirq) begin
        if (irq_pending) begin
          latched_store <= 1;
          reg_out <= irq_pending;
          reg_next_pc <= current_pc + (compressed_instr ? 2 : 4);
          mem_do_rinst <= 1;
        end else
          do_waitirq <= 1;
      end else
      if (decoder_trigger) begin
        `debug($display("-- %-0t", $time);)
        irq_delay <= irq_active;
        reg_next_pc <= current_pc + (compressed_instr ? 2 : 4);
        if (ENABLE_TRACE)
          latched_trace <= 1;
        if (ENABLE_COUNTERS) begin
          count_instr <= count_instr + 1;
          if (!ENABLE_COUNTERS64) count_instr[63:32] <= 0;
        end
        if (instr_jal) begin
          mem_do_rinst <= 1;
          reg_next_pc <= current_pc + decoded_imm_j;
          latched_branch <= 1;
        end else begin
          mem_do_rinst <= 0;
          mem_do_prefetch <= !instr_jalr && !instr_retirq;
          cpu_state <= cpu_state_ld_rs1;
        end
      end
    } .elsewhen( cpu_state_ld_rs1 === cpu_state ){
      reg_op1 <= 'bx;
      reg_op2 <= 'bx;

      (* parallel_case *)
      case (1'b1)
        (CATCH_ILLINSN || WITH_PCPI) && instr_trap: begin
          if (WITH_PCPI) begin
            `debug($display("LD_RS1: %2d 0x%08x", decoded_rs1, cpuregs_rs1);)
            reg_op1 <= cpuregs_rs1;
            dbg_rs1val <= cpuregs_rs1;
            dbg_rs1val_valid <= 1;
            if (ENABLE_REGS_DUALPORT) begin
              pcpi_valid <= 1;
              `debug($display("LD_RS2: %2d 0x%08x", decoded_rs2, cpuregs_rs2);)
              reg_sh <= cpuregs_rs2;
              reg_op2 <= cpuregs_rs2;
              dbg_rs2val <= cpuregs_rs2;
              dbg_rs2val_valid <= 1;
              if (pcpi_int_ready) begin
                mem_do_rinst <= 1;
                pcpi_valid <= 0;
                reg_out <= pcpi_int_rd;
                latched_store <= pcpi_int_wr;
                cpu_state <= cpu_state_fetch;
              end else
              if (CATCH_ILLINSN && (pcpi_timeout || instr_ecall_ebreak)) begin
                pcpi_valid <= 0;
                `debug($display("EBREAK OR UNSUPPORTED INSN AT 0x%08x", reg_pc);)
                if (ENABLE_IRQ && !irq_mask[irq_ebreak] && !irq_active) begin
                  next_irq_pending[irq_ebreak] = 1;
                  cpu_state <= cpu_state_fetch;
                end else
                  cpu_state <= cpu_state_trap;
              end
            end else begin
              cpu_state <= cpu_state_ld_rs2;
            end
          end else begin
            `debug($display("EBREAK OR UNSUPPORTED INSN AT 0x%08x", reg_pc);)
            if (ENABLE_IRQ && !irq_mask[irq_ebreak] && !irq_active) begin
              next_irq_pending[irq_ebreak] = 1;
              cpu_state <= cpu_state_fetch;
            end else
              cpu_state <= cpu_state_trap;
          end
        end
        ENABLE_COUNTERS && is_rdcycle_rdcycleh_rdinstr_rdinstrh: begin
          (* parallel_case, full_case *)
          case (1'b1)
            instr_rdcycle:
              reg_out <= count_cycle[31:0];
            instr_rdcycleh && ENABLE_COUNTERS64:
              reg_out <= count_cycle[63:32];
            instr_rdinstr:
              reg_out <= count_instr[31:0];
            instr_rdinstrh && ENABLE_COUNTERS64:
              reg_out <= count_instr[63:32];
          endcase
          latched_store <= 1;
          cpu_state <= cpu_state_fetch;
        end
        is_lui_auipc_jal: begin
          reg_op1 <= instr_lui ? 0 : reg_pc;
          reg_op2 <= decoded_imm;
          if (TWO_CYCLE_ALU)
            alu_wait <= 1;
          else
            mem_do_rinst <= mem_do_prefetch;
          cpu_state <= cpu_state_exec;
        end
        ENABLE_IRQ && ENABLE_IRQ_QREGS && instr_getq: begin
          `debug($display("LD_RS1: %2d 0x%08x", decoded_rs1, cpuregs_rs1);)
          reg_out <= cpuregs_rs1;
          dbg_rs1val <= cpuregs_rs1;
          dbg_rs1val_valid <= 1;
          latched_store <= 1;
          cpu_state <= cpu_state_fetch;
        end
        ENABLE_IRQ && ENABLE_IRQ_QREGS && instr_setq: begin
          `debug($display("LD_RS1: %2d 0x%08x", decoded_rs1, cpuregs_rs1);)
          reg_out <= cpuregs_rs1;
          dbg_rs1val <= cpuregs_rs1;
          dbg_rs1val_valid <= 1;
          latched_rd <= latched_rd | irqregs_offset;
          latched_store <= 1;
          cpu_state <= cpu_state_fetch;
        end
        ENABLE_IRQ && instr_retirq: begin
          eoi <= 0;
          irq_active <= 0;
          latched_branch <= 1;
          latched_store <= 1;
          `debug($display("LD_RS1: %2d 0x%08x", decoded_rs1, cpuregs_rs1);)
          reg_out <= CATCH_MISALIGN ? (cpuregs_rs1 & 32'h fffffffe) : cpuregs_rs1;
          dbg_rs1val <= cpuregs_rs1;
          dbg_rs1val_valid <= 1;
          cpu_state <= cpu_state_fetch;
        end
        ENABLE_IRQ && instr_maskirq: begin
          latched_store <= 1;
          reg_out <= irq_mask;
          `debug($display("LD_RS1: %2d 0x%08x", decoded_rs1, cpuregs_rs1);)
          irq_mask <= cpuregs_rs1 | MASKED_IRQ;
          dbg_rs1val <= cpuregs_rs1;
          dbg_rs1val_valid <= 1;
          cpu_state <= cpu_state_fetch;
        end
        ENABLE_IRQ && ENABLE_IRQ_TIMER && instr_timer: begin
          latched_store <= 1;
          reg_out <= timer;
          `debug($display("LD_RS1: %2d 0x%08x", decoded_rs1, cpuregs_rs1);)
          timer <= cpuregs_rs1;
          dbg_rs1val <= cpuregs_rs1;
          dbg_rs1val_valid <= 1;
          cpu_state <= cpu_state_fetch;
        end
        is_lb_lh_lw_lbu_lhu && !instr_trap: begin
          `debug($display("LD_RS1: %2d 0x%08x", decoded_rs1, cpuregs_rs1);)
          reg_op1 <= cpuregs_rs1;
          dbg_rs1val <= cpuregs_rs1;
          dbg_rs1val_valid <= 1;
          cpu_state <= cpu_state_ldmem;
          mem_do_rinst <= 1;
        end
        is_slli_srli_srai && !BARREL_SHIFTER: begin
          `debug($display("LD_RS1: %2d 0x%08x", decoded_rs1, cpuregs_rs1);)
          reg_op1 <= cpuregs_rs1;
          dbg_rs1val <= cpuregs_rs1;
          dbg_rs1val_valid <= 1;
          reg_sh <= decoded_rs2;
          cpu_state <= cpu_state_shift;
        end
        is_jalr_addi_slti_sltiu_xori_ori_andi, is_slli_srli_srai && BARREL_SHIFTER: begin
          `debug($display("LD_RS1: %2d 0x%08x", decoded_rs1, cpuregs_rs1);)
          reg_op1 <= cpuregs_rs1;
          dbg_rs1val <= cpuregs_rs1;
          dbg_rs1val_valid <= 1;
          reg_op2 <= is_slli_srli_srai && BARREL_SHIFTER ? decoded_rs2 : decoded_imm;
          if (TWO_CYCLE_ALU)
            alu_wait <= 1;
          else
            mem_do_rinst <= mem_do_prefetch;
          cpu_state <= cpu_state_exec;
        end
        default: begin
          `debug($display("LD_RS1: %2d 0x%08x", decoded_rs1, cpuregs_rs1);)
          reg_op1 <= cpuregs_rs1;
          dbg_rs1val <= cpuregs_rs1;
          dbg_rs1val_valid <= 1;
          if (ENABLE_REGS_DUALPORT) begin
            `debug($display("LD_RS2: %2d 0x%08x", decoded_rs2, cpuregs_rs2);)
            reg_sh <= cpuregs_rs2;
            reg_op2 <= cpuregs_rs2;
            dbg_rs2val <= cpuregs_rs2;
            dbg_rs2val_valid <= 1;
            (* parallel_case *)
            case (1'b1)
              is_sb_sh_sw: begin
                cpu_state <= cpu_state_stmem;
                mem_do_rinst <= 1;
              end
              is_sll_srl_sra && !BARREL_SHIFTER: begin
                cpu_state <= cpu_state_shift;
              end
              default: begin
                if (TWO_CYCLE_ALU || (TWO_CYCLE_COMPARE && is_beq_bne_blt_bge_bltu_bgeu)) begin
                  alu_wait_2 <= TWO_CYCLE_ALU && (TWO_CYCLE_COMPARE && is_beq_bne_blt_bge_bltu_bgeu);
                  alu_wait <= 1;
                end else
                  mem_do_rinst <= mem_do_prefetch;
                cpu_state <= cpu_state_exec;
              end
            endcase
          end else
            cpu_state <= cpu_state_ld_rs2;
        end
      endcase
    } .elsewhen( cpu_state_ld_rs2 === cpu_state ){
      `debug($display("LD_RS2: %2d 0x%08x", decoded_rs2, cpuregs_rs2);)
      reg_sh <= cpuregs_rs2;
      reg_op2 <= cpuregs_rs2;
      dbg_rs2val <= cpuregs_rs2;
      dbg_rs2val_valid <= 1;

      (* parallel_case *)
      case (1'b1)
        WITH_PCPI && instr_trap: begin
          pcpi_valid <= 1;
          if (pcpi_int_ready) begin
            mem_do_rinst <= 1;
            pcpi_valid <= 0;
            reg_out <= pcpi_int_rd;
            latched_store <= pcpi_int_wr;
            cpu_state <= cpu_state_fetch;
          end else
          if (CATCH_ILLINSN && (pcpi_timeout || instr_ecall_ebreak)) begin
            pcpi_valid <= 0;
            `debug($display("EBREAK OR UNSUPPORTED INSN AT 0x%08x", reg_pc);)
            if (ENABLE_IRQ && !irq_mask[irq_ebreak] && !irq_active) begin
              next_irq_pending[irq_ebreak] = 1;
              cpu_state <= cpu_state_fetch;
            end else
              cpu_state <= cpu_state_trap;
          end
        end
        is_sb_sh_sw: begin
          cpu_state <= cpu_state_stmem;
          mem_do_rinst <= 1;
        end
        is_sll_srl_sra && !BARREL_SHIFTER: begin
          cpu_state <= cpu_state_shift;
        end
        default: begin
          if (TWO_CYCLE_ALU || (TWO_CYCLE_COMPARE && is_beq_bne_blt_bge_bltu_bgeu)) begin
            alu_wait_2 <= TWO_CYCLE_ALU && (TWO_CYCLE_COMPARE && is_beq_bne_blt_bge_bltu_bgeu);
            alu_wait <= 1;
          end else
            mem_do_rinst <= mem_do_prefetch;
          cpu_state <= cpu_state_exec;
        end
      endcase
    } .elsewhen( cpu_state_exec  === cpu_state  ){
      reg_out <= reg_pc + decoded_imm;
      if ((TWO_CYCLE_ALU || TWO_CYCLE_COMPARE) && (alu_wait || alu_wait_2)) begin
        mem_do_rinst <= mem_do_prefetch && !alu_wait_2;
        alu_wait <= alu_wait_2;
      end else
      if (is_beq_bne_blt_bge_bltu_bgeu) begin
        latched_rd <= 0;
        latched_store <= TWO_CYCLE_COMPARE ? alu_out_0_q : alu_out_0;
        latched_branch <= TWO_CYCLE_COMPARE ? alu_out_0_q : alu_out_0;
        if (mem_done)
          cpu_state <= cpu_state_fetch;
        if (TWO_CYCLE_COMPARE ? alu_out_0_q : alu_out_0) begin
          decoder_trigger <= 0;
          set_mem_do_rinst = 1;
        end
      end else begin
        latched_branch <= instr_jalr;
        latched_store <= 1;
        latched_stalu <= 1;
        cpu_state <= cpu_state_fetch;
      end        
    } .elsewhen( cpu_state_shift  === cpu_state ){
      latched_store <= 1;
      if (reg_sh == 0) begin
        reg_out <= reg_op1;
        mem_do_rinst <= mem_do_prefetch;
        cpu_state <= cpu_state_fetch;
      end else if (TWO_STAGE_SHIFT && reg_sh >= 4) begin
        (* parallel_case, full_case *)
        case (1'b1)
          instr_slli || instr_sll: reg_op1 <= reg_op1 << 4;
          instr_srli || instr_srl: reg_op1 <= reg_op1 >> 4;
          instr_srai || instr_sra: reg_op1 <= $signed(reg_op1) >>> 4;
        endcase
        reg_sh <= reg_sh - 4;
      end else begin
        (* parallel_case, full_case *)
        case (1'b1)
          instr_slli || instr_sll: reg_op1 <= reg_op1 << 1;
          instr_srli || instr_srl: reg_op1 <= reg_op1 >> 1;
          instr_srai || instr_sra: reg_op1 <= $signed(reg_op1) >>> 1;
        endcase
        reg_sh <= reg_sh - 1;
      end        
    } .elsewhen( cpu_state_stmem === cpu_state ){
      if (ENABLE_TRACE)
        reg_out <= reg_op2;
      if (!mem_do_prefetch || mem_done) begin
        if (!mem_do_wdata) begin
          (* parallel_case, full_case *)
          case (1'b1)
            instr_sb: mem_wordsize <= 2;
            instr_sh: mem_wordsize <= 1;
            instr_sw: mem_wordsize <= 0;
          endcase
          if (ENABLE_TRACE) begin
            trace_valid <= 1;
            trace_data <= (irq_active ? TRACE_IRQ : 0) | TRACE_ADDR | ((reg_op1 + decoded_imm) & 32'hffffffff);
          end
          reg_op1 <= reg_op1 + decoded_imm;
          set_mem_do_wdata = 1;
        end
        if (!mem_do_prefetch && mem_done) begin
          cpu_state <= cpu_state_fetch;
          decoder_trigger <= 1;
          decoder_pseudo_trigger <= 1;
        end
      end
    } .elsewhen( cpu_state_ldmem === cpu_state ){
      latched_store <= 1;
      if (!mem_do_prefetch || mem_done) begin
        if (!mem_do_rdata) begin
          (* parallel_case, full_case *)
          case (1'b1)
            instr_lb || instr_lbu: mem_wordsize <= 2;
            instr_lh || instr_lhu: mem_wordsize <= 1;
            instr_lw: mem_wordsize <= 0;
          endcase
          latched_is_lu <= is_lbu_lhu_lw;
          latched_is_lh <= instr_lh;
          latched_is_lb <= instr_lb;
          if (ENABLE_TRACE) begin
            trace_valid <= 1;
            trace_data <= (irq_active ? TRACE_IRQ : 0) | TRACE_ADDR | ((reg_op1 + decoded_imm) & 32'hffffffff);
          end
          reg_op1 <= reg_op1 + decoded_imm;
          set_mem_do_rdata = 1;
        end
        if (!mem_do_prefetch && mem_done) begin
          (* parallel_case, full_case *)
          case (1'b1)
            latched_is_lu: reg_out <= mem_rdata_word;
            latched_is_lh: reg_out <= $signed(mem_rdata_word[15:0]);
            latched_is_lb: reg_out <= $signed(mem_rdata_word[7:0]);
          endcase
          decoder_trigger <= 1;
          decoder_pseudo_trigger <= 1;
          cpu_state <= cpu_state_fetch;
        end
      end        
    }






		if (ENABLE_IRQ){
			next_irq_pending = next_irq_pending | irq;
			if(ENABLE_IRQ_TIMER){
        when( timer === 1.U ){
          next_irq_pending[irq_timer] = 1;   
        }
      }
    }


		if (CATCH_MISALIGN && resetn && (mem_do_rdata || mem_do_wdata)) begin
			if (mem_wordsize == 0 && reg_op1[1:0] != 0) begin
				`debug($display("MISALIGNED WORD: 0x%08x", reg_op1);)
				if (ENABLE_IRQ && !irq_mask[irq_buserror] && !irq_active) begin
					next_irq_pending[irq_buserror] = 1;
				end else
					cpu_state <= cpu_state_trap;
			end
			if (mem_wordsize == 1 && reg_op1[0] != 0) begin
				`debug($display("MISALIGNED HALFWORD: 0x%08x", reg_op1);)
				if (ENABLE_IRQ && !irq_mask[irq_buserror] && !irq_active) begin
					next_irq_pending[irq_buserror] = 1;
				end else
					cpu_state <= cpu_state_trap;
			end
		end
		if (CATCH_MISALIGN && resetn && mem_do_rinst && (COMPRESSED_ISA ? reg_pc[0] : |reg_pc[1:0])) begin
			`debug($display("MISALIGNED INSTRUCTION: 0x%08x", reg_pc);)
			if (ENABLE_IRQ && !irq_mask[irq_buserror] && !irq_active) begin
				next_irq_pending[irq_buserror] = 1;
			end else
				cpu_state <= cpu_state_trap;
		end
		if (!CATCH_ILLINSN && decoder_trigger_q && !decoder_pseudo_trigger_q && instr_ecall_ebreak) begin
			cpu_state <= cpu_state_trap;
		end

		if (!resetn || mem_done) begin
			mem_do_prefetch <= 0;
			mem_do_rinst <= 0;
			mem_do_rdata <= 0;
			mem_do_wdata <= 0;
		end

		if (set_mem_do_rinst)
			mem_do_rinst <= 1;
		if (set_mem_do_rdata)
			mem_do_rdata <= 1;
		if (set_mem_do_wdata)
			mem_do_wdata <= 1;

		irq_pending <= next_irq_pending & ~MASKED_IRQ;

		if (!CATCH_MISALIGN) begin
			if (COMPRESSED_ISA) begin
				reg_pc[0] <= 0;
				reg_next_pc[0] <= 0;
			end else begin
				reg_pc[1:0] <= 0;
				reg_next_pc[1:0] <= 0;
			end
		end
		current_pc = 'bx;
	end

`ifdef RISCV_FORMAL
	reg dbg_irq_call;
	reg dbg_irq_enter;
	reg [31:0] dbg_irq_ret;
	always @(posedge clk) begin
		rvfi_valid <= resetn && (launch_next_insn || trap) && dbg_valid_insn;
		rvfi_order <= resetn ? rvfi_order + rvfi_valid : 0;

		rvfi_insn <= dbg_insn_opcode;
		rvfi_rs1_addr <= dbg_rs1val_valid ? dbg_insn_rs1 : 0;
		rvfi_rs2_addr <= dbg_rs2val_valid ? dbg_insn_rs2 : 0;
		rvfi_pc_rdata <= dbg_insn_addr;
		rvfi_rs1_rdata <= dbg_rs1val_valid ? dbg_rs1val : 0;
		rvfi_rs2_rdata <= dbg_rs2val_valid ? dbg_rs2val : 0;
		rvfi_trap <= trap;
		rvfi_halt <= trap;
		rvfi_intr <= dbg_irq_enter;
		rvfi_mode <= 3;
		rvfi_ixl <= 1;

		if (!resetn) begin
			dbg_irq_call <= 0;
			dbg_irq_enter <= 0;
		end else
		if (rvfi_valid) begin
			dbg_irq_call <= 0;
			dbg_irq_enter <= dbg_irq_call;
		end else
		if (irq_state == 1) begin
			dbg_irq_call <= 1;
			dbg_irq_ret <= next_pc;
		end

		if (!resetn) begin
			rvfi_rd_addr <= 0;
			rvfi_rd_wdata <= 0;
		end else
		if (cpuregs_write && !irq_state) begin
`ifdef PICORV32_TESTBUG_003
			rvfi_rd_addr <= latched_rd ^ 1;
`else
			rvfi_rd_addr <= latched_rd;
`endif
`ifdef PICORV32_TESTBUG_004
			rvfi_rd_wdata <= latched_rd ? cpuregs_wrdata ^ 1 : 0;
`else
			rvfi_rd_wdata <= latched_rd ? cpuregs_wrdata : 0;
`endif
		end else
		if (rvfi_valid) begin
			rvfi_rd_addr <= 0;
			rvfi_rd_wdata <= 0;
		end

		casez (dbg_insn_opcode)
			32'b 0000000_?????_000??_???_?????_0001011: begin // getq
				rvfi_rs1_addr <= 0;
				rvfi_rs1_rdata <= 0;
			end
			32'b 0000001_?????_?????_???_000??_0001011: begin // setq
				rvfi_rd_addr <= 0;
				rvfi_rd_wdata <= 0;
			end
			32'b 0000010_?????_00000_???_00000_0001011: begin // retirq
				rvfi_rs1_addr <= 0;
				rvfi_rs1_rdata <= 0;
			end
		endcase

		if (!dbg_irq_call) begin
			if (dbg_mem_instr) begin
				rvfi_mem_addr <= 0;
				rvfi_mem_rmask <= 0;
				rvfi_mem_wmask <= 0;
				rvfi_mem_rdata <= 0;
				rvfi_mem_wdata <= 0;
			end else
			if (dbg_mem_valid && dbg_mem_ready) begin
				rvfi_mem_addr <= dbg_mem_addr;
				rvfi_mem_rmask <= dbg_mem_wstrb ? 0 : ~0;
				rvfi_mem_wmask <= dbg_mem_wstrb;
				rvfi_mem_rdata <= dbg_mem_rdata;
				rvfi_mem_wdata <= dbg_mem_wdata;
			end
		end
	end





}




module picorv32 #(

) (


);




// `ifndef PICORV32_REGS
// 	reg [31:0] cpuregs [0:regfile_size-1];

// 	integer i;
// 	initial begin
// 		if (REGS_INIT_ZERO) begin
// 			for (i = 0; i < regfile_size; i = i+1)
// 				cpuregs[i] = 0;
// 		end
// 	end
// `endif

// `ifdef DEBUGREGS
// 	val dbg_reg_x0  = 0
// 	val dbg_reg_x1  = cpuregs[1]
// 	val dbg_reg_x2  = cpuregs[2]
// 	val dbg_reg_x3  = cpuregs[3]
// 	val dbg_reg_x4  = cpuregs[4]
// 	val dbg_reg_x5  = cpuregs[5]
// 	val dbg_reg_x6  = cpuregs[6]
// 	val dbg_reg_x7  = cpuregs[7]
// 	val dbg_reg_x8  = cpuregs[8]
// 	val dbg_reg_x9  = cpuregs[9]
// 	val dbg_reg_x10 = cpuregs[10]
// 	val dbg_reg_x11 = cpuregs[11]
// 	val dbg_reg_x12 = cpuregs[12]
// 	val dbg_reg_x13 = cpuregs[13]
// 	val dbg_reg_x14 = cpuregs[14]
// 	val dbg_reg_x15 = cpuregs[15]
// 	val dbg_reg_x16 = cpuregs[16]
// 	val dbg_reg_x17 = cpuregs[17]
// 	val dbg_reg_x18 = cpuregs[18]
// 	val dbg_reg_x19 = cpuregs[19]
// 	val dbg_reg_x20 = cpuregs[20]
// 	val dbg_reg_x21 = cpuregs[21]
// 	val dbg_reg_x22 = cpuregs[22]
// 	val dbg_reg_x23 = cpuregs[23]
// 	val dbg_reg_x24 = cpuregs[24]
// 	val dbg_reg_x25 = cpuregs[25]
// 	val dbg_reg_x26 = cpuregs[26]
// 	val dbg_reg_x27 = cpuregs[27]
// 	val dbg_reg_x28 = cpuregs[28]
// 	val dbg_reg_x29 = cpuregs[29]
// 	val dbg_reg_x30 = cpuregs[30]
// 	val dbg_reg_x31 = cpuregs[31]
// `endif



	// reg [63:0] new_ascii_instr;


	// always @* begin
	// 	new_ascii_instr = "";

	// 	if (instr_lui)      new_ascii_instr = "lui";
	// 	if (instr_auipc)    new_ascii_instr = "auipc";
	// 	if (instr_jal)      new_ascii_instr = "jal";
	// 	if (instr_jalr)     new_ascii_instr = "jalr";

	// 	if (instr_beq)      new_ascii_instr = "beq";
	// 	if (instr_bne)      new_ascii_instr = "bne";
	// 	if (instr_blt)      new_ascii_instr = "blt";
	// 	if (instr_bge)      new_ascii_instr = "bge";
	// 	if (instr_bltu)     new_ascii_instr = "bltu";
	// 	if (instr_bgeu)     new_ascii_instr = "bgeu";

	// 	if (instr_lb)       new_ascii_instr = "lb";
	// 	if (instr_lh)       new_ascii_instr = "lh";
	// 	if (instr_lw)       new_ascii_instr = "lw";
	// 	if (instr_lbu)      new_ascii_instr = "lbu";
	// 	if (instr_lhu)      new_ascii_instr = "lhu";
	// 	if (instr_sb)       new_ascii_instr = "sb";
	// 	if (instr_sh)       new_ascii_instr = "sh";
	// 	if (instr_sw)       new_ascii_instr = "sw";

	// 	if (instr_addi)     new_ascii_instr = "addi";
	// 	if (instr_slti)     new_ascii_instr = "slti";
	// 	if (instr_sltiu)    new_ascii_instr = "sltiu";
	// 	if (instr_xori)     new_ascii_instr = "xori";
	// 	if (instr_ori)      new_ascii_instr = "ori";
	// 	if (instr_andi)     new_ascii_instr = "andi";
	// 	if (instr_slli)     new_ascii_instr = "slli";
	// 	if (instr_srli)     new_ascii_instr = "srli";
	// 	if (instr_srai)     new_ascii_instr = "srai";

	// 	if (instr_add)      new_ascii_instr = "add";
	// 	if (instr_sub)      new_ascii_instr = "sub";
	// 	if (instr_sll)      new_ascii_instr = "sll";
	// 	if (instr_slt)      new_ascii_instr = "slt";
	// 	if (instr_sltu)     new_ascii_instr = "sltu";
	// 	if (instr_xor)      new_ascii_instr = "xor";
	// 	if (instr_srl)      new_ascii_instr = "srl";
	// 	if (instr_sra)      new_ascii_instr = "sra";
	// 	if (instr_or)       new_ascii_instr = "or";
	// 	if (instr_and)      new_ascii_instr = "and";

	// 	if (instr_rdcycle)  new_ascii_instr = "rdcycle";
	// 	if (instr_rdcycleh) new_ascii_instr = "rdcycleh";
	// 	if (instr_rdinstr)  new_ascii_instr = "rdinstr";
	// 	if (instr_rdinstrh) new_ascii_instr = "rdinstrh";
	// 	if (instr_fence)    new_ascii_instr = "fence";

	// 	if (instr_getq)     new_ascii_instr = "getq";
	// 	if (instr_setq)     new_ascii_instr = "setq";
	// 	if (instr_retirq)   new_ascii_instr = "retirq";
	// 	if (instr_maskirq)  new_ascii_instr = "maskirq";
	// 	if (instr_waitirq)  new_ascii_instr = "waitirq";
	// 	if (instr_timer)    new_ascii_instr = "timer";
	// end


	`FORMAL_KEEP reg [127:0] dbg_ascii_state;

	always @* begin
		dbg_ascii_state = "";
		if (cpu_state == cpu_state_trap)   dbg_ascii_state = "trap";
		if (cpu_state == cpu_state_fetch)  dbg_ascii_state = "fetch";
		if (cpu_state == cpu_state_ld_rs1) dbg_ascii_state = "ld_rs1";
		if (cpu_state == cpu_state_ld_rs2) dbg_ascii_state = "ld_rs2";
		if (cpu_state == cpu_state_exec)   dbg_ascii_state = "exec";
		if (cpu_state == cpu_state_shift)  dbg_ascii_state = "shift";
		if (cpu_state == cpu_state_stmem)  dbg_ascii_state = "stmem";
		if (cpu_state == cpu_state_ldmem)  dbg_ascii_state = "ldmem";
	end



	always @* begin
`ifdef PICORV32_TESTBUG_005
		rvfi_pc_wdata = (dbg_irq_call ? dbg_irq_ret : dbg_insn_addr) ^ 4;
`else
		rvfi_pc_wdata = dbg_irq_call ? dbg_irq_ret : dbg_insn_addr;
`endif

		rvfi_csr_mcycle_rmask = 0;
		rvfi_csr_mcycle_wmask = 0;
		rvfi_csr_mcycle_rdata = 0;
		rvfi_csr_mcycle_wdata = 0;

		rvfi_csr_minstret_rmask = 0;
		rvfi_csr_minstret_wmask = 0;
		rvfi_csr_minstret_rdata = 0;
		rvfi_csr_minstret_wdata = 0;

		if (rvfi_valid && rvfi_insn[6:0] == 7'b 1110011 && rvfi_insn[13:12] == 3'b010) begin
			if (rvfi_insn[31:20] == 12'h C00) begin
				rvfi_csr_mcycle_rmask = 64'h 0000_0000_FFFF_FFFF;
				rvfi_csr_mcycle_rdata = {32'h 0000_0000, rvfi_rd_wdata};
			end
			if (rvfi_insn[31:20] == 12'h C80) begin
				rvfi_csr_mcycle_rmask = 64'h FFFF_FFFF_0000_0000;
				rvfi_csr_mcycle_rdata = {rvfi_rd_wdata, 32'h 0000_0000};
			end
			if (rvfi_insn[31:20] == 12'h C02) begin
				rvfi_csr_minstret_rmask = 64'h 0000_0000_FFFF_FFFF;
				rvfi_csr_minstret_rdata = {32'h 0000_0000, rvfi_rd_wdata};
			end
			if (rvfi_insn[31:20] == 12'h C82) begin
				rvfi_csr_minstret_rmask = 64'h FFFF_FFFF_0000_0000;
				rvfi_csr_minstret_rdata = {rvfi_rd_wdata, 32'h 0000_0000};
			end
		end
	end
`endif

	// Formal Verification
`ifdef FORMAL
	reg [3:0] last_mem_nowait;
	always @(posedge clk)
		last_mem_nowait <= {last_mem_nowait, io.mem.ready || !mem_valid};

	// stall the memory interface for max 4 cycles
	restrict property (|last_mem_nowait || io.mem.ready || !mem_valid);

	// resetn low in first cycle, after that resetn high
	restrict property (resetn != $initstate);

	// this just makes it much easier to read traces. uncomment as needed.
	// assume property (mem_valid || !mem_ready);

	reg ok;
	always @* begin
		if (resetn) begin
			// instruction fetches are read-only
			if (mem_valid && mem_instr)
				assert (mem_wstrb == 0);

			// cpu_state must be valid
			ok = 0;
			if (cpu_state == cpu_state_trap)   ok = 1;
			if (cpu_state == cpu_state_fetch)  ok = 1;
			if (cpu_state == cpu_state_ld_rs1) ok = 1;
			if (cpu_state == cpu_state_ld_rs2) ok = !ENABLE_REGS_DUALPORT;
			if (cpu_state == cpu_state_exec)   ok = 1;
			if (cpu_state == cpu_state_shift)  ok = 1;
			if (cpu_state == cpu_state_stmem)  ok = 1;
			if (cpu_state == cpu_state_ldmem)  ok = 1;
			assert (ok);
		end
	end

	reg last_mem_la_read = 0;
	reg last_mem_la_write = 0;
	reg [31:0] last_mem_la_addr;
	reg [31:0] last_mem_la_wdata;
	reg [3:0] last_mem_la_wstrb = 0;

	always @(posedge clk) begin
		last_mem_la_read <= mem_la_read;
		last_mem_la_write <= mem_la_write;
		last_mem_la_addr <= mem_la_addr;
		last_mem_la_wdata <= mem_la_wdata;
		last_mem_la_wstrb <= mem_la_wstrb;

		if (last_mem_la_read) begin
			assert(mem_valid);
			assert(mem_addr == last_mem_la_addr);
			assert(mem_wstrb == 0);
		end
		if (last_mem_la_write) begin
			assert(mem_valid);
			assert(mem_addr == last_mem_la_addr);
			assert(mem_wdata == last_mem_la_wdata);
			assert(mem_wstrb == last_mem_la_wstrb);
		end
		if (mem_la_read || mem_la_write) begin
			assert(!mem_valid || io.mem.ready);
		end
	end
`endif
endmodule

// This is a simple example implementation of PICORV32_REGS.
// Use the PICORV32_REGS mechanism if you want to use custom
// memory resources to implement the processor register file.
// Note that your implementation must match the requirements of
// the PicoRV32 configuration. (e.g. QREGS, etc)
module picorv32_regs (
	input clk, wen,
	input [5:0] waddr,
	input [5:0] raddr1,
	input [5:0] raddr2,
	input [31:0] wdata,
	output [31:0] rdata1,
	output [31:0] rdata2
);
	reg [31:0] regs [0:30];

	always @(posedge clk)
		if (wen) regs[~waddr[4:0]] <= wdata;

	assign rdata1 = regs[~raddr1[4:0]];
	assign rdata2 = regs[~raddr2[4:0]];
endmodule









/***************************************************************
 * picorv32_axi
 ***************************************************************/

module picorv32_axi #(
	parameter [ 0:0] ENABLE_COUNTERS = 1,
	parameter [ 0:0] ENABLE_COUNTERS64 = 1,
	parameter [ 0:0] ENABLE_REGS_16_31 = 1,
	parameter [ 0:0] ENABLE_REGS_DUALPORT = 1,
	parameter [ 0:0] TWO_STAGE_SHIFT = 1,
	parameter [ 0:0] BARREL_SHIFTER = 0,
	parameter [ 0:0] TWO_CYCLE_COMPARE = 0,
	parameter [ 0:0] TWO_CYCLE_ALU = 0,
	parameter [ 0:0] COMPRESSED_ISA = 0,
	parameter [ 0:0] CATCH_MISALIGN = 1,
	parameter [ 0:0] CATCH_ILLINSN = 1,
	parameter [ 0:0] ENABLE_PCPI = 0,
	parameter [ 0:0] ENABLE_MUL = 0,
	parameter [ 0:0] ENABLE_FAST_MUL = 0,
	parameter [ 0:0] ENABLE_DIV = 0,
	parameter [ 0:0] ENABLE_IRQ = 0,
	parameter [ 0:0] ENABLE_IRQ_QREGS = 1,
	parameter [ 0:0] ENABLE_IRQ_TIMER = 1,
	parameter [ 0:0] ENABLE_TRACE = 0,
	parameter [ 0:0] REGS_INIT_ZERO = 0,
	parameter [31:0] MASKED_IRQ = 32'h 0000_0000,
	parameter [31:0] LATCHED_IRQ = 32'h ffff_ffff,
	parameter [31:0] PROGADDR_RESET = 32'h 0000_0000,
	parameter [31:0] PROGADDR_IRQ = 32'h 0000_0010,
	parameter [31:0] STACKADDR = 32'h ffff_ffff
) (
	input clk, resetn,
	output trap,

	// AXI4-lite master memory interface

	output        mem_axi_awvalid,
	input         mem_axi_awready,
	output [31:0] mem_axi_awaddr,
	output [ 2:0] mem_axi_awprot,

	output        mem_axi_wvalid,
	input         mem_axi_wready,
	output [31:0] mem_axi_wdata,
	output [ 3:0] mem_axi_wstrb,

	input         mem_axi_bvalid,
	output        mem_axi_bready,

	output        mem_axi_arvalid,
	input         mem_axi_arready,
	output [31:0] mem_axi_araddr,
	output [ 2:0] mem_axi_arprot,

	input         mem_axi_rvalid,
	output        mem_axi_rready,
	input  [31:0] mem_axi_rdata,

	// Pico Co-Processor Interface (PCPI)
	output        pcpi_valid,
	output [31:0] pcpi_insn,
	output [31:0] pcpi_rs1,
	output [31:0] pcpi_rs2,
	input         pcpi_wr,
	input  [31:0] pcpi_rd,
	input         pcpi_wait,
	input         pcpi_ready,

	// IRQ interface
	input  [31:0] irq,
	output [31:0] eoi,

`ifdef RISCV_FORMAL
	output        rvfi_valid,
	output [63:0] rvfi_order,
	output [31:0] rvfi_insn,
	output        rvfi_trap,
	output        rvfi_halt,
	output        rvfi_intr,
	output [ 4:0] rvfi_rs1_addr,
	output [ 4:0] rvfi_rs2_addr,
	output [31:0] rvfi_rs1_rdata,
	output [31:0] rvfi_rs2_rdata,
	output [ 4:0] rvfi_rd_addr,
	output [31:0] rvfi_rd_wdata,
	output [31:0] rvfi_pc_rdata,
	output [31:0] rvfi_pc_wdata,
	output [31:0] rvfi_mem_addr,
	output [ 3:0] rvfi_mem_rmask,
	output [ 3:0] rvfi_mem_wmask,
	output [31:0] rvfi_mem_rdata,
	output [31:0] rvfi_mem_wdata,
`endif

	// Trace Interface
	output        trace_valid,
	output [35:0] trace_data
);
	wire        mem_valid;
	wire [31:0] mem_addr;
	wire [31:0] mem_wdata;
	wire [ 3:0] mem_wstrb;
	wire        mem_instr;
	wire        mem_ready;
	wire [31:0] mem_rdata;

	picorv32_axi_adapter axi_adapter (
		.clk            (clk            ),
		.resetn         (resetn         ),
		.mem_axi_awvalid(mem_axi_awvalid),
		.mem_axi_awready(mem_axi_awready),
		.mem_axi_awaddr (mem_axi_awaddr ),
		.mem_axi_awprot (mem_axi_awprot ),
		.mem_axi_wvalid (mem_axi_wvalid ),
		.mem_axi_wready (mem_axi_wready ),
		.mem_axi_wdata  (mem_axi_wdata  ),
		.mem_axi_wstrb  (mem_axi_wstrb  ),
		.mem_axi_bvalid (mem_axi_bvalid ),
		.mem_axi_bready (mem_axi_bready ),
		.mem_axi_arvalid(mem_axi_arvalid),
		.mem_axi_arready(mem_axi_arready),
		.mem_axi_araddr (mem_axi_araddr ),
		.mem_axi_arprot (mem_axi_arprot ),
		.mem_axi_rvalid (mem_axi_rvalid ),
		.mem_axi_rready (mem_axi_rready ),
		.mem_axi_rdata  (mem_axi_rdata  ),
		.mem_valid      (mem_valid      ),
		.mem_instr      (mem_instr      ),
		.mem_ready      (mem_ready      ),
		.mem_addr       (mem_addr       ),
		.mem_wdata      (mem_wdata      ),
		.mem_wstrb      (mem_wstrb      ),
		.mem_rdata      (mem_rdata      )
	);

	picorv32 #(
		.ENABLE_COUNTERS     (ENABLE_COUNTERS     ),
		.ENABLE_COUNTERS64   (ENABLE_COUNTERS64   ),
		.ENABLE_REGS_16_31   (ENABLE_REGS_16_31   ),
		.ENABLE_REGS_DUALPORT(ENABLE_REGS_DUALPORT),
		.TWO_STAGE_SHIFT     (TWO_STAGE_SHIFT     ),
		.BARREL_SHIFTER      (BARREL_SHIFTER      ),
		.TWO_CYCLE_COMPARE   (TWO_CYCLE_COMPARE   ),
		.TWO_CYCLE_ALU       (TWO_CYCLE_ALU       ),
		.COMPRESSED_ISA      (COMPRESSED_ISA      ),
		.CATCH_MISALIGN      (CATCH_MISALIGN      ),
		.CATCH_ILLINSN       (CATCH_ILLINSN       ),
		.ENABLE_PCPI         (ENABLE_PCPI         ),
		.ENABLE_MUL          (ENABLE_MUL          ),
		.ENABLE_FAST_MUL     (ENABLE_FAST_MUL     ),
		.ENABLE_DIV          (ENABLE_DIV          ),
		.ENABLE_IRQ          (ENABLE_IRQ          ),
		.ENABLE_IRQ_QREGS    (ENABLE_IRQ_QREGS    ),
		.ENABLE_IRQ_TIMER    (ENABLE_IRQ_TIMER    ),
		.ENABLE_TRACE        (ENABLE_TRACE        ),
		.REGS_INIT_ZERO      (REGS_INIT_ZERO      ),
		.MASKED_IRQ          (MASKED_IRQ          ),
		.LATCHED_IRQ         (LATCHED_IRQ         ),
		.PROGADDR_RESET      (PROGADDR_RESET      ),
		.PROGADDR_IRQ        (PROGADDR_IRQ        ),
		.STACKADDR           (STACKADDR           )
	) picorv32_core (
		.clk      (clk   ),
		.resetn   (resetn),
		.trap     (trap  ),

		.mem_valid(mem_valid),
		.mem_addr (mem_addr ),
		.mem_wdata(mem_wdata),
		.mem_wstrb(mem_wstrb),
		.mem_instr(mem_instr),
		.mem_ready(mem_ready),
		.mem_rdata(mem_rdata),

		.pcpi_valid(pcpi_valid),
		.pcpi_insn (pcpi_insn ),
		.pcpi_rs1  (pcpi_rs1  ),
		.pcpi_rs2  (pcpi_rs2  ),
		.pcpi_wr   (pcpi_wr   ),
		.pcpi_rd   (pcpi_rd   ),
		.pcpi_wait (pcpi_wait ),
		.pcpi_ready(pcpi_ready),

		.irq(irq),
		.eoi(eoi),

`ifdef RISCV_FORMAL
		.rvfi_valid    (rvfi_valid    ),
		.rvfi_order    (rvfi_order    ),
		.rvfi_insn     (rvfi_insn     ),
		.rvfi_trap     (rvfi_trap     ),
		.rvfi_halt     (rvfi_halt     ),
		.rvfi_intr     (rvfi_intr     ),
		.rvfi_rs1_addr (rvfi_rs1_addr ),
		.rvfi_rs2_addr (rvfi_rs2_addr ),
		.rvfi_rs1_rdata(rvfi_rs1_rdata),
		.rvfi_rs2_rdata(rvfi_rs2_rdata),
		.rvfi_rd_addr  (rvfi_rd_addr  ),
		.rvfi_rd_wdata (rvfi_rd_wdata ),
		.rvfi_pc_rdata (rvfi_pc_rdata ),
		.rvfi_pc_wdata (rvfi_pc_wdata ),
		.rvfi_mem_addr (rvfi_mem_addr ),
		.rvfi_mem_rmask(rvfi_mem_rmask),
		.rvfi_mem_wmask(rvfi_mem_wmask),
		.rvfi_mem_rdata(rvfi_mem_rdata),
		.rvfi_mem_wdata(rvfi_mem_wdata),
`endif

		.trace_valid(trace_valid),
		.trace_data (trace_data)
	);
endmodule


/***************************************************************
 * picorv32_axi_adapter
 ***************************************************************/

module picorv32_axi_adapter (
	input clk, resetn,

	// AXI4-lite master memory interface

	output        mem_axi_awvalid,
	input         mem_axi_awready,
	output [31:0] mem_axi_awaddr,
	output [ 2:0] mem_axi_awprot,

	output        mem_axi_wvalid,
	input         mem_axi_wready,
	output [31:0] mem_axi_wdata,
	output [ 3:0] mem_axi_wstrb,

	input         mem_axi_bvalid,
	output        mem_axi_bready,

	output        mem_axi_arvalid,
	input         mem_axi_arready,
	output [31:0] mem_axi_araddr,
	output [ 2:0] mem_axi_arprot,

	input         mem_axi_rvalid,
	output        mem_axi_rready,
	input  [31:0] mem_axi_rdata,

	// Native PicoRV32 memory interface

	input         mem_valid,
	input         mem_instr,
	output        mem_ready,
	input  [31:0] mem_addr,
	input  [31:0] mem_wdata,
	input  [ 3:0] mem_wstrb,
	output [31:0] mem_rdata
);
	reg ack_awvalid;
	reg ack_arvalid;
	reg ack_wvalid;
	reg xfer_done;

	assign mem_axi_awvalid = mem_valid && |mem_wstrb && !ack_awvalid;
	assign mem_axi_awaddr = mem_addr;
	assign mem_axi_awprot = 0;

	assign mem_axi_arvalid = mem_valid && !mem_wstrb && !ack_arvalid;
	assign mem_axi_araddr = mem_addr;
	assign mem_axi_arprot = mem_instr ? 3'b100 : 3'b000;

	assign mem_axi_wvalid = mem_valid && |mem_wstrb && !ack_wvalid;
	assign mem_axi_wdata = mem_wdata;
	assign mem_axi_wstrb = mem_wstrb;

	assign mem_ready = mem_axi_bvalid || mem_axi_rvalid;
	assign mem_axi_bready = mem_valid && |mem_wstrb;
	assign mem_axi_rready = mem_valid && !mem_wstrb;
	assign mem_rdata = mem_axi_rdata;

	always @(posedge clk) begin
		if (!resetn) begin
			ack_awvalid <= 0;
		end else begin
			xfer_done <= mem_valid && mem_ready;
			if (mem_axi_awready && mem_axi_awvalid)
				ack_awvalid <= 1;
			if (mem_axi_arready && mem_axi_arvalid)
				ack_arvalid <= 1;
			if (mem_axi_wready && mem_axi_wvalid)
				ack_wvalid <= 1;
			if (xfer_done || !mem_valid) begin
				ack_awvalid <= 0;
				ack_arvalid <= 0;
				ack_wvalid <= 0;
			end
		end
	end
endmodule


/***************************************************************
 * picorv32_wb
 ***************************************************************/

module picorv32_wb #(
	parameter [ 0:0] ENABLE_COUNTERS = 1,
	parameter [ 0:0] ENABLE_COUNTERS64 = 1,
	parameter [ 0:0] ENABLE_REGS_16_31 = 1,
	parameter [ 0:0] ENABLE_REGS_DUALPORT = 1,
	parameter [ 0:0] TWO_STAGE_SHIFT = 1,
	parameter [ 0:0] BARREL_SHIFTER = 0,
	parameter [ 0:0] TWO_CYCLE_COMPARE = 0,
	parameter [ 0:0] TWO_CYCLE_ALU = 0,
	parameter [ 0:0] COMPRESSED_ISA = 0,
	parameter [ 0:0] CATCH_MISALIGN = 1,
	parameter [ 0:0] CATCH_ILLINSN = 1,
	parameter [ 0:0] ENABLE_PCPI = 0,
	parameter [ 0:0] ENABLE_MUL = 0,
	parameter [ 0:0] ENABLE_FAST_MUL = 0,
	parameter [ 0:0] ENABLE_DIV = 0,
	parameter [ 0:0] ENABLE_IRQ = 0,
	parameter [ 0:0] ENABLE_IRQ_QREGS = 1,
	parameter [ 0:0] ENABLE_IRQ_TIMER = 1,
	parameter [ 0:0] ENABLE_TRACE = 0,
	parameter [ 0:0] REGS_INIT_ZERO = 0,
	parameter [31:0] MASKED_IRQ = 32'h 0000_0000,
	parameter [31:0] LATCHED_IRQ = 32'h ffff_ffff,
	parameter [31:0] PROGADDR_RESET = 32'h 0000_0000,
	parameter [31:0] PROGADDR_IRQ = 32'h 0000_0010,
	parameter [31:0] STACKADDR = 32'h ffff_ffff
) (
	output trap,

	// Wishbone interfaces
	input wb_rst_i,
	input wb_clk_i,

	output reg [31:0] wbm_adr_o,
	output reg [31:0] wbm_dat_o,
	input [31:0] wbm_dat_i,
	output reg wbm_we_o,
	output reg [3:0] wbm_sel_o,
	output reg wbm_stb_o,
	input wbm_ack_i,
	output reg wbm_cyc_o,

	// Pico Co-Processor Interface (PCPI)
	output        pcpi_valid,
	output [31:0] pcpi_insn,
	output [31:0] pcpi_rs1,
	output [31:0] pcpi_rs2,
	input         pcpi_wr,
	input  [31:0] pcpi_rd,
	input         pcpi_wait,
	input         pcpi_ready,

	// IRQ interface
	input  [31:0] irq,
	output [31:0] eoi,

`ifdef RISCV_FORMAL
	output        rvfi_valid,
	output [63:0] rvfi_order,
	output [31:0] rvfi_insn,
	output        rvfi_trap,
	output        rvfi_halt,
	output        rvfi_intr,
	output [ 4:0] rvfi_rs1_addr,
	output [ 4:0] rvfi_rs2_addr,
	output [31:0] rvfi_rs1_rdata,
	output [31:0] rvfi_rs2_rdata,
	output [ 4:0] rvfi_rd_addr,
	output [31:0] rvfi_rd_wdata,
	output [31:0] rvfi_pc_rdata,
	output [31:0] rvfi_pc_wdata,
	output [31:0] rvfi_mem_addr,
	output [ 3:0] rvfi_mem_rmask,
	output [ 3:0] rvfi_mem_wmask,
	output [31:0] rvfi_mem_rdata,
	output [31:0] rvfi_mem_wdata,
`endif

	// Trace Interface
	output        trace_valid,
	output [35:0] trace_data,

	output mem_instr
);
	wire        mem_valid;
	wire [31:0] mem_addr;
	wire [31:0] mem_wdata;
	wire [ 3:0] mem_wstrb;
	reg         mem_ready;
	reg [31:0] mem_rdata;

	wire clk;
	wire resetn;

	assign clk = wb_clk_i;
	assign resetn = ~wb_rst_i;

	picorv32 #(
		.ENABLE_COUNTERS     (ENABLE_COUNTERS     ),
		.ENABLE_COUNTERS64   (ENABLE_COUNTERS64   ),
		.ENABLE_REGS_16_31   (ENABLE_REGS_16_31   ),
		.ENABLE_REGS_DUALPORT(ENABLE_REGS_DUALPORT),
		.TWO_STAGE_SHIFT     (TWO_STAGE_SHIFT     ),
		.BARREL_SHIFTER      (BARREL_SHIFTER      ),
		.TWO_CYCLE_COMPARE   (TWO_CYCLE_COMPARE   ),
		.TWO_CYCLE_ALU       (TWO_CYCLE_ALU       ),
		.COMPRESSED_ISA      (COMPRESSED_ISA      ),
		.CATCH_MISALIGN      (CATCH_MISALIGN      ),
		.CATCH_ILLINSN       (CATCH_ILLINSN       ),
		.ENABLE_PCPI         (ENABLE_PCPI         ),
		.ENABLE_MUL          (ENABLE_MUL          ),
		.ENABLE_FAST_MUL     (ENABLE_FAST_MUL     ),
		.ENABLE_DIV          (ENABLE_DIV          ),
		.ENABLE_IRQ          (ENABLE_IRQ          ),
		.ENABLE_IRQ_QREGS    (ENABLE_IRQ_QREGS    ),
		.ENABLE_IRQ_TIMER    (ENABLE_IRQ_TIMER    ),
		.ENABLE_TRACE        (ENABLE_TRACE        ),
		.REGS_INIT_ZERO      (REGS_INIT_ZERO      ),
		.MASKED_IRQ          (MASKED_IRQ          ),
		.LATCHED_IRQ         (LATCHED_IRQ         ),
		.PROGADDR_RESET      (PROGADDR_RESET      ),
		.PROGADDR_IRQ        (PROGADDR_IRQ        ),
		.STACKADDR           (STACKADDR           )
	) picorv32_core (
		.clk      (clk   ),
		.resetn   (resetn),
		.trap     (trap  ),

		.mem_valid(mem_valid),
		.mem_addr (mem_addr ),
		.mem_wdata(mem_wdata),
		.mem_wstrb(mem_wstrb),
		.mem_instr(mem_instr),
		.mem_ready(mem_ready),
		.mem_rdata(mem_rdata),

		.pcpi_valid(pcpi_valid),
		.pcpi_insn (pcpi_insn ),
		.pcpi_rs1  (pcpi_rs1  ),
		.pcpi_rs2  (pcpi_rs2  ),
		.pcpi_wr   (pcpi_wr   ),
		.pcpi_rd   (pcpi_rd   ),
		.pcpi_wait (pcpi_wait ),
		.pcpi_ready(pcpi_ready),

		.irq(irq),
		.eoi(eoi),

`ifdef RISCV_FORMAL
		.rvfi_valid    (rvfi_valid    ),
		.rvfi_order    (rvfi_order    ),
		.rvfi_insn     (rvfi_insn     ),
		.rvfi_trap     (rvfi_trap     ),
		.rvfi_halt     (rvfi_halt     ),
		.rvfi_intr     (rvfi_intr     ),
		.rvfi_rs1_addr (rvfi_rs1_addr ),
		.rvfi_rs2_addr (rvfi_rs2_addr ),
		.rvfi_rs1_rdata(rvfi_rs1_rdata),
		.rvfi_rs2_rdata(rvfi_rs2_rdata),
		.rvfi_rd_addr  (rvfi_rd_addr  ),
		.rvfi_rd_wdata (rvfi_rd_wdata ),
		.rvfi_pc_rdata (rvfi_pc_rdata ),
		.rvfi_pc_wdata (rvfi_pc_wdata ),
		.rvfi_mem_addr (rvfi_mem_addr ),
		.rvfi_mem_rmask(rvfi_mem_rmask),
		.rvfi_mem_wmask(rvfi_mem_wmask),
		.rvfi_mem_rdata(rvfi_mem_rdata),
		.rvfi_mem_wdata(rvfi_mem_wdata),
`endif

		.trace_valid(trace_valid),
		.trace_data (trace_data)
	);

	localparam IDLE = 2'b00;
	localparam WBSTART = 2'b01;
	localparam WBEND = 2'b10;

	reg [1:0] state;

	wire we;
	assign we = (mem_wstrb[0] | mem_wstrb[1] | mem_wstrb[2] | mem_wstrb[3]);

	always @(posedge wb_clk_i) begin
		if (wb_rst_i) begin
			wbm_adr_o <= 0;
			wbm_dat_o <= 0;
			wbm_we_o <= 0;
			wbm_sel_o <= 0;
			wbm_stb_o <= 0;
			wbm_cyc_o <= 0;
			state <= IDLE;
		end else begin
			case (state)
				IDLE: begin
					if (mem_valid) begin
						wbm_adr_o <= mem_addr;
						wbm_dat_o <= mem_wdata;
						wbm_we_o <= we;
						wbm_sel_o <= mem_wstrb;

						wbm_stb_o <= 1'b1;
						wbm_cyc_o <= 1'b1;
						state <= WBSTART;
					end else begin
						mem_ready <= 1'b0;

						wbm_stb_o <= 1'b0;
						wbm_cyc_o <= 1'b0;
						wbm_we_o <= 1'b0;
					end
				end
				WBSTART:begin
					if (wbm_ack_i) begin
						mem_rdata <= wbm_dat_i;
						mem_ready <= 1'b1;

						state <= WBEND;

						wbm_stb_o <= 1'b0;
						wbm_cyc_o <= 1'b0;
						wbm_we_o <= 1'b0;
					end
				end
				WBEND: begin
					mem_ready <= 1'b0;

					state <= IDLE;
				end
				default:
					state <= IDLE;
			endcase
		end
	end
endmodule
