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
  val wstrb = Output(UInt(4.W))
  val rdata = Input(UInt(32.W))

}

class Memory_LookAHead_Bundle extends Bundle{
  val read  = Bool()
  val write = Bool()
  val addr  = UInt(32.W)
  val wdata = UInt(32.W)
  val wstrb = UInt(4.W)
}


class Riscv_Format_Bundle extends Bundle{
  val valid = Bool()
  val order = UInt(64.W)
  val insn  = UInt(32.W)
  val trap  = Bool()
  val halt  = Bool()
  val intr  = Bool()
  val mode  = UInt(2.W)
  val ixl   = UInt(2.W)
  val rs1_addr  = UInt(5.W)
  val rs2_addr  = UInt(5.W)
  val rs1_rdata = UInt(32.W)
  val rs2_rdata = UInt(32.W)
  val rd_addr   = UInt(5.W)
  val rd_wdata  = UInt(32.W)
  val pc_rdata  = UInt(32.W)
  val pc_wdata  = UInt(32.W)
  val mem_addr  = UInt(32.W)
  val mem_rmask = UInt(4.W)
  val mem_wmask = UInt(4.W)
  val mem_rdata = UInt(32.W)
  val mem_wdata = UInt(32.W)

  val csr_mcycle_rmask = UInt(64.W)
  val csr_mcycle_wmask = UInt(64.W)
  val csr_mcycle_rdata = UInt(64.W)
  val csr_mcycle_wdata = UInt(64.W)

  val csr_minstret_rmask= UInt(64.W)
  val csr_minstret_wmask= UInt(64.W)
  val csr_minstret_rdata= UInt(64.W)
  val csr_minstret_wdata= UInt(64.W)
}


class Picorv32(
  ENABLE_COUNTERS: Boolean = true,
  ENABLE_COUNTERS64: Boolean = true,
  ENABLE_REGS_16_31: Boolean = true,
  ENABLE_REGS_DUALPORT: Boolean = true,
  LATCHED_MEM_RDATA: Boolean = false,
  TWO_CYCLE_COMPARE: Boolean = true,
  TWO_CYCLE_ALU: Boolean = true,
  CATCH_MISALIGN: Boolean = true,
  CATCH_ILLINSN: Boolean = true,
  ENABLE_IRQ: Boolean = true,
  ENABLE_IRQ_QREGS: Boolean = true,
  ENABLE_IRQ_TIMER: Boolean = true,
  ENABLE_TRACE: Boolean = true,
  MASKED_IRQ: UInt = "h00000000".U(32.W),
  LATCHED_IRQ: UInt = "hffffffff".U(32.W),
  PROGADDR_RESET: UInt = "h00000000".U(32.W),
  PROGADDR_IRQ: UInt = "h00000010".U(32.W),
  STACKADDR: UInt = "hffffffff".U(32.W),
)
extends Module{
  def RISCV_FORMAL = true


  class Picorv32IO extends Bundle{
  val trap = Output(Bool())
  val mem  = new Memory_Access_Bundle
  val mem_la = Output(new Memory_LookAHead_Bundle)

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





  val eoi = RegInit(0.U(32.W)); io.eoi := eoi

  val trace_valid = Reg(Bool()); io.trace.valid := trace_valid
  val trace_data = Reg(UInt(36.W)); io.trace.bits := trace_data



  def irq_timer    = 0
  def irq_ebreak   = 1
  def irq_buserror = 2

  def irqregs_offset = if(ENABLE_REGS_16_31) {32.U} else {16.U}
  def regfile_size   = if(ENABLE_REGS_16_31) { if(ENABLE_IRQ & ENABLE_IRQ_QREGS) {36} else {32} } else { if(ENABLE_IRQ & ENABLE_IRQ_QREGS) {20} else {16} }
  def regindex_bits  = if(ENABLE_REGS_16_31) { if(ENABLE_IRQ & ENABLE_IRQ_QREGS) { 6} else { 5} } else { if(ENABLE_IRQ & ENABLE_IRQ_QREGS) { 5} else { 4} }


  def TRACE_BRANCH = Cat( "b0001".U(4.W), 0.U(32.W) )
  def TRACE_ADDR   = Cat( "b0010".U(4.W), 0.U(32.W) )
  def TRACE_IRQ    = Cat( "b1000".U(4.W), 0.U(32.W) )


  val count_cycle = RegInit(0.U( (if(ENABLE_COUNTERS64){64} else{32}).W) )      
  if (ENABLE_COUNTERS){
    count_cycle := count_cycle + 1.U
  }

  val count_instr = if(ENABLE_COUNTERS64 ) { RegInit(0.U(64.W)) } else { RegInit(0.U(32.W)) }

  val reg_pc      = RegInit(PROGADDR_RESET)
  val reg_next_pc = RegInit(PROGADDR_RESET)
  val reg_op1 = Reg(UInt(32.W))
  val reg_op2 = Reg(UInt(32.W))
  val reg_out = if( STACKADDR != "hffffffff".U(32.W) ) { RegInit(STACKADDR) } else { Reg(UInt(32.W)) }


  val dbg_mem_valid = mem_valid
  val dbg_mem_instr = mem_instr
  val dbg_mem_ready = io.mem.ready

  val dbg_mem_addr  = mem_addr
  val dbg_mem_wdata = mem_wdata
  val dbg_mem_wstrb = mem_wstrb
  val dbg_mem_rdata = io.mem.rdata



  val next_pc = Wire(UInt(32.W))

  val irq_delay = RegInit(false.B)
  val irq_active = RegInit(false.B)
  val irq_mask = RegInit("hFFFFFFFF".U(32.W))

  val irq_pending = RegInit(0.U(32.W))
  val timer = RegInit(0.U(32.W))

















  // Memory Interface
  
  val mem_wordsize = Reg(UInt(2.W))


  io.mem_la.wdata := 
    Mux1H(Seq(
      ( mem_wordsize === 0.U ) -> reg_op2,
      ( mem_wordsize === 1.U ) -> Fill(2, reg_op2(15,0)),
      ( mem_wordsize === 2.U ) -> Fill(4, reg_op2( 7,0)),
    ))

  io.mem_la.wstrb := 
    Mux1H(Seq(
      ( mem_wordsize === 0.U ) -> "b1111".U,
      ( mem_wordsize === 1.U ) -> Mux( reg_op1.extract(1), "b1100".U, "b0011".U),
      ( mem_wordsize === 2.U ) -> (1.U << reg_op1(1,0)),
    ))

  val mem_rdata_word = 
    Mux1H(Seq(
      ( mem_wordsize === 0.U ) -> io.mem.rdata,
      ( mem_wordsize === 1.U ) -> Mux( reg_op1.extract(1), io.mem.rdata(31,16), io.mem.rdata(15, 0) ),
      ( mem_wordsize === 2.U ) ->
        Mux1H(Seq(
          (reg_op1(1,0) === "b00".U) -> io.mem.rdata( 7, 0),
          (reg_op1(1,0) === "b01".U) -> io.mem.rdata(15, 8),
          (reg_op1(1,0) === "b10".U) -> io.mem.rdata(23,16),
          (reg_op1(1,0) === "b11".U) -> io.mem.rdata(31,24),
        )),
    ))




  val mem_state = RegInit(0.U(2.W))



  val mem_do_prefetch = RegInit(false.B)
  val mem_do_rinst = RegInit(false.B)
  val mem_do_rdata = RegInit(false.B)
  val mem_do_wdata = RegInit(false.B)

  val mem_xfer = mem_valid & io.mem.ready


  val mem_rdata_q = RegNext(io.mem.rdata)

  val mem_rdata_latched = (if( LATCHED_MEM_RDATA ){ io.mem.rdata } else { Mux(mem_xfer, io.mem.rdata, mem_rdata_q) })

  val mem_busy = mem_do_prefetch | mem_do_rinst | mem_do_rdata | mem_do_wdata

  val mem_done = ~reset.asBool & ((mem_xfer & mem_state.orR & (mem_do_rinst | mem_do_rdata | mem_do_wdata)) | (mem_state.andR & mem_do_rinst))

  io.mem_la.write := ~reset.asBool & (mem_state === 0.U) & mem_do_wdata
  io.mem_la.read  := ~reset.asBool & (mem_state === 0.U) & (mem_do_rinst | mem_do_prefetch | mem_do_rdata)

  io.mem_la.addr := Mux( mem_do_prefetch | mem_do_rinst, Cat(next_pc(31,2), 0.U(2.W)), Cat(reg_op1(31,2), 0.U(2.W)))


  val next_insn_opcode = RegEnable( io.mem.rdata, mem_xfer)







  when(~reset.asBool & ~io.trap) {
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

  } .otherwise{
    when( io.mem_la.read | io.mem_la.write) {
      mem_addr  := io.mem_la.addr
      mem_wstrb := io.mem_la.wstrb & Fill(4, io.mem_la.write)
    }
    when(io.mem_la.write) {
      mem_wdata := io.mem_la.wdata
    }

    when( mem_state === 0.U ){
      when(mem_do_prefetch | mem_do_rinst | mem_do_rdata){
        mem_valid := true.B
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
      assert(mem_valid === true.B)
      assert(mem_instr === (mem_do_prefetch | mem_do_rinst))
      when(mem_xfer){
        mem_valid := false.B
        mem_state := Mux(mem_do_rinst | mem_do_rdata, 0.U, 3.U)
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
  val instr_getq = Wire(Bool())
  val instr_setq = Wire(Bool())
  val instr_retirq = Wire(Bool())
  val instr_maskirq = Wire(Bool())
  val instr_waitirq = Wire(Bool())
  val instr_timer = Wire(Bool())



  val decoded_rd  = Reg(UInt(regindex_bits.W))
  val decoded_rs1 = Reg(UInt(regindex_bits.W))
  val decoded_rs2 = Reg(UInt(5.W))
  val decoded_rs  = Wire(UInt(regindex_bits.W))

  val decoded_imm = Reg(UInt(32.W))
  val decoded_imm_j = Reg(UInt(32.W))

  val decoder_trigger  = Reg(Bool())
  val decoder_trigger_q  = RegNext(decoder_trigger)
  val decoder_pseudo_trigger  = Reg(Bool())
  val decoder_pseudo_trigger_q  = RegNext(decoder_pseudo_trigger)

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
    if( CATCH_ILLINSN ){
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

  val dbg_insn_imm    = Wire( UInt(32.W) ); dontTouch(dbg_insn_imm) 
  val dbg_insn_rs1    = Wire( UInt(5.W) );  dontTouch(dbg_insn_rs1) 
  val dbg_insn_rs2    = Wire( UInt(5.W) );  dontTouch(dbg_insn_rs2) 
  val dbg_insn_rd     = Wire( UInt(5.W) );  dontTouch(dbg_insn_rd) 

  val dbg_rs1val = Reg(UInt(32.W)); dontTouch(dbg_rs1val)
  val dbg_rs2val = Reg(UInt(32.W)); dontTouch(dbg_rs2val)
  val dbg_rs1val_valid = Reg(Bool()); dontTouch(dbg_rs1val_valid)
  val dbg_rs2val_valid = Reg(Bool()); dontTouch(dbg_rs2val_valid)

  val launch_next_insn = Wire(Bool())
  val dbg_valid_insn = RegInit(false.B)

  val q_insn_imm    = RegNext( dbg_insn_imm )
  val q_insn_opcode = RegNext( dbg_insn_opcode )
  val q_insn_rs1    = RegNext( dbg_insn_rs1 )
  val q_insn_rs2    = RegNext( dbg_insn_rs2 )
  val q_insn_rd     = RegNext( dbg_insn_rd )
  val dbg_next      = RegNext( launch_next_insn )



  when(io.trap){
    dbg_valid_insn := false.B
  } .elsewhen(launch_next_insn){
    dbg_valid_insn := true.B
  }


  val cached_insn_imm    = RegEnable(decoded_imm, decoder_trigger_q)
  val cached_insn_opcode = RegEnable( Mux( next_insn_opcode(1,0) === "b11".U, next_insn_opcode, next_insn_opcode(15,0) ), decoder_trigger_q)
  val cached_insn_rs1 = RegEnable(decoded_rs1, decoder_trigger_q)
  val cached_insn_rs2 = RegEnable(decoded_rs2, decoder_trigger_q)
  val cached_insn_rd  = RegEnable(decoded_rd,  decoder_trigger_q)

  val dbg_insn_addr  = RegEnable(next_pc, launch_next_insn)


  dbg_insn_imm    := Mux( dbg_next, Mux( decoder_pseudo_trigger_q, cached_insn_imm, decoded_imm), q_insn_imm )
  dbg_insn_opcode := Mux( dbg_next, Mux( decoder_pseudo_trigger_q, cached_insn_opcode, Mux( next_insn_opcode(1,0) === "b11".U, next_insn_opcode, next_insn_opcode(15,0) ) ), q_insn_opcode)
  dbg_insn_rs1    := Mux( dbg_next, Mux( decoder_pseudo_trigger_q, cached_insn_rs1, decoded_rs1 ), q_insn_rs1 )
  dbg_insn_rs2    := Mux( dbg_next, Mux( decoder_pseudo_trigger_q, cached_insn_rs2, decoded_rs2 ), q_insn_rs2 )
  dbg_insn_rd     := Mux( dbg_next, Mux( decoder_pseudo_trigger_q, cached_insn_rd , decoded_rd  ), q_insn_rd  )


  if(ENABLE_IRQ) {
    instr_retirq  := RegEnable( mem_rdata_latched === BitPat("b0000010??????????????????0001011"), mem_do_rinst & mem_done)
    instr_waitirq := RegEnable( mem_rdata_latched === BitPat("b0000100??????????????????0001011"), mem_do_rinst & mem_done)
  } else {
    instr_retirq  := false.B
    instr_waitirq := false.B
  }

  when(mem_do_rinst & mem_done){
    instr_lui     := mem_rdata_latched(6,0) === "b0110111".U
    instr_auipc   := mem_rdata_latched(6,0) === "b0010111".U
    instr_jal     := mem_rdata_latched(6,0) === "b1101111".U
    instr_jalr    := mem_rdata_latched === BitPat("b?????????????????000?????1100111")


    is_beq_bne_blt_bge_bltu_bgeu := mem_rdata_latched(6,0) === "b1100011".U
    is_lb_lh_lw_lbu_lhu          := mem_rdata_latched(6,0) === "b0000011".U
    is_sb_sh_sw                  := mem_rdata_latched(6,0) === "b0100011".U
    is_alu_reg_imm               := mem_rdata_latched(6,0) === "b0010011".U
    is_alu_reg_reg               := mem_rdata_latched(6,0) === "b0110011".U

    {
      val temp = Cat( Fill(11, mem_rdata_latched.extract(31) ), mem_rdata_latched(31,12), 0.U(1.W) )
      decoded_imm_j := Cat( temp(31,20), temp(8,1), temp.extract(9), temp(19,10), temp.extract(0) )
    }


    decoded_rd  := mem_rdata_latched(11,7)
    decoded_rs1 := mem_rdata_latched(19,15)
    decoded_rs2 := mem_rdata_latched(24,20)

    if(ENABLE_IRQ && ENABLE_IRQ_QREGS){
      when( mem_rdata_latched === BitPat("b0000000??????????????????0001011") ){
        decoded_rs1 := Cat( 1.U(1.W), mem_rdata_latched(19,15) ) // instr_getq
      }        
    }

    if(ENABLE_IRQ){
      when(mem_rdata_latched === BitPat("b0000010??????????????????0001011") ){
        decoded_rs1 := (if(ENABLE_IRQ_QREGS) {irqregs_offset} else {3.U}) // instr_retirq
      }
    }

  }


    if( ENABLE_IRQ ){
      instr_maskirq := RegEnable(mem_rdata_q === BitPat("b0000011??????????????????0001011"), decoder_trigger & ~decoder_pseudo_trigger)
    } else {
      instr_maskirq := false.B
    }

    if( ENABLE_IRQ & ENABLE_IRQ_QREGS ){
      instr_getq    := RegEnable(mem_rdata_q === BitPat("b0000000??????????????????0001011"), decoder_trigger & ~decoder_pseudo_trigger)
      instr_setq    := RegEnable(mem_rdata_q === BitPat("b0000001??????????????????0001011"), decoder_trigger & ~decoder_pseudo_trigger)
      instr_timer   := RegEnable(mem_rdata_q === BitPat("b0000101??????????????????0001011"), decoder_trigger & ~decoder_pseudo_trigger)
    } else {
      instr_getq    := false.B
      instr_setq    := false.B
      instr_timer   := false.B
    }




  when(decoder_trigger & ~decoder_pseudo_trigger){

    instr_beq   := is_beq_bne_blt_bge_bltu_bgeu & mem_rdata_q(14,12) === "b000".U
    instr_bne   := is_beq_bne_blt_bge_bltu_bgeu & mem_rdata_q(14,12) === "b001".U
    instr_blt   := is_beq_bne_blt_bge_bltu_bgeu & mem_rdata_q(14,12) === "b100".U
    instr_bge   := is_beq_bne_blt_bge_bltu_bgeu & mem_rdata_q(14,12) === "b101".U
    instr_bltu  := is_beq_bne_blt_bge_bltu_bgeu & mem_rdata_q(14,12) === "b110".U
    instr_bgeu  := is_beq_bne_blt_bge_bltu_bgeu & mem_rdata_q(14,12) === "b111".U

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

    instr_slli  := is_alu_reg_imm & mem_rdata_q === BitPat("b0000000??????????001????????????")
    instr_srli  := is_alu_reg_imm & mem_rdata_q === BitPat("b0000000??????????101????????????")
    instr_srai  := is_alu_reg_imm & mem_rdata_q === BitPat("b0100000??????????101????????????")
    instr_add   := is_alu_reg_reg & mem_rdata_q === BitPat("b0000000??????????000????????????")
    instr_sub   := is_alu_reg_reg & mem_rdata_q === BitPat("b0100000??????????000????????????")
    instr_sll   := is_alu_reg_reg & mem_rdata_q === BitPat("b0000000??????????001????????????")
    instr_slt   := is_alu_reg_reg & mem_rdata_q === BitPat("b0000000??????????010????????????")
    instr_sltu  := is_alu_reg_reg & mem_rdata_q === BitPat("b0000000??????????011????????????")
    instr_xor   := is_alu_reg_reg & mem_rdata_q === BitPat("b0000000??????????100????????????")
    instr_srl   := is_alu_reg_reg & mem_rdata_q === BitPat("b0000000??????????101????????????")
    instr_sra   := is_alu_reg_reg & mem_rdata_q === BitPat("b0100000??????????101????????????")
    instr_or    := is_alu_reg_reg & mem_rdata_q === BitPat("b0000000??????????110????????????")
    instr_and   := is_alu_reg_reg & mem_rdata_q === BitPat("b0000000??????????111????????????")

    if( ENABLE_COUNTERS ){
      instr_rdcycle := mem_rdata_q === BitPat("b11000000000?00000010?????1110011")
      instr_rdinstr := mem_rdata_q === BitPat("b11000000001000000010?????1110011")

      if( ENABLE_COUNTERS64 ){
        instr_rdcycleh := mem_rdata_q === BitPat("b11001000000?00000010?????1110011")
        instr_rdinstrh := mem_rdata_q === BitPat("b11001000001000000010?????1110011")
      } else {
        instr_rdcycleh := false.B
        instr_rdinstrh := false.B
      }
    } else {
      instr_rdcycle  := false.B
      instr_rdinstr  := false.B
      instr_rdcycleh := false.B
      instr_rdinstrh := false.B
    }


    instr_ecall_ebreak := mem_rdata_q === BitPat("b00000000000?00000000000001110011")
    instr_fence := mem_rdata_q === BitPat("b?????????????????000?????0001111")





    is_slli_srli_srai := is_alu_reg_imm & (
      mem_rdata_q === BitPat("b0000000??????????001????????????") |
      mem_rdata_q === BitPat("b0000000??????????101????????????") |
      mem_rdata_q === BitPat("b0100000??????????101????????????")
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
      mem_rdata_q === BitPat("b0000000??????????001????????????") |
      mem_rdata_q === BitPat("b0000000??????????101????????????") |
      mem_rdata_q === BitPat("b0100000??????????101????????????")    
    )

    decoded_imm := Mux1H(Seq(
      instr_jal                                           -> decoded_imm_j,
      (instr_lui | instr_auipc)                           -> (mem_rdata_q(31,12) << 12),
      (instr_jalr | is_lb_lh_lw_lbu_lhu | is_alu_reg_imm) -> Cat( Fill( 20, mem_rdata_q.extract(31)), mem_rdata_q(31,20) ),
      is_beq_bne_blt_bge_bltu_bgeu                        -> Cat( Fill( 19, mem_rdata_q.extract(31)), mem_rdata_q.extract(31), mem_rdata_q.extract(7), mem_rdata_q(30,25), mem_rdata_q(11,8), 0.U(1.W)),
      is_sb_sh_sw                                         -> Cat( Fill( 20, mem_rdata_q.extract(31)), mem_rdata_q(31,25), mem_rdata_q(11,7)),
    ))
  }











































  // Main State Machine

  val cpu_state_trap   = "b10000000".U(8.W)
  val cpu_state_fetch  = "b01000000".U(8.W)
  val cpu_state_ld_rs1 = "b00100000".U(8.W)
  val cpu_state_ld_rs2 = "b00010000".U(8.W)
  val cpu_state_exec   = "b00001000".U(8.W)
  // val cpu_state_shift  = "b00000100".U(8.W)
  val cpu_state_stmem  = "b00000010".U(8.W)
  val cpu_state_ldmem  = "b00000001".U(8.W)

  val cpu_state = RegInit(cpu_state_fetch)
  val irq_state = RegInit(0.U(2.W))



  val latched_store  = RegInit((if(STACKADDR != "hffffffff".U(32.W)){true.B}else{false.B}))
  val latched_stalu  = RegInit(false.B)
  val latched_branch = RegInit(false.B)
  val latched_trace = RegInit(false.B)
  val latched_is_lu = RegInit(false.B)
  val latched_is_lh = RegInit(false.B)
  val latched_is_lb = RegInit(false.B)

  val latched_rd = if(STACKADDR != "hffffffff".U(32.W)){RegInit(2.U(regindex_bits.W))} else { Reg(UInt((regindex_bits.W))) }

  next_pc := Mux(latched_store & latched_branch, reg_out & "hfffffffe".U, reg_next_pc)



  val do_waitirq = Reg(Bool())


  val alu_wait = Reg(Bool())
  val alu_wait_2 = Reg(Bool())

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
    (if( !TWO_CYCLE_COMPARE ){
      Seq(
        is_slti_blt_slt     -> alu_lts,
        is_sltiu_bltu_sltu  -> alu_ltu,      
      )
    } else { Seq() })
  )

  val alu_out = Mux1H(Seq(
      is_lui_auipc_jal_jalr_addi_add_sub -> alu_add_sub,
      is_compare -> alu_out_0,
      ( instr_xori | instr_xor ) -> ( reg_op1 ^ reg_op2 ),
      ( instr_ori  | instr_or  ) -> ( reg_op1 | reg_op2 ),
      ( instr_andi | instr_and ) -> ( reg_op1 & reg_op2 ),
      (instr_sll | instr_slli)   -> alu_shl,
      (instr_srl | instr_srli | instr_sra | instr_srai) -> alu_shr,      
    )
  )
  dontTouch(alu_out)
  val alu_out_q = RegNext(alu_out)
  val alu_out_0_q = RegNext(alu_out_0)





  val cpuregs_write = 
    (cpu_state === cpu_state_fetch) & (
      latched_branch | latched_store |
      (if( ENABLE_IRQ ) { irq_state.extract(0) | irq_state.extract(1) } else { false.B })
    )

  val cpuregs_wrdata =
    Mux1H(Seq(
        ((cpu_state === cpu_state_fetch) & latched_branch                 ) -> (reg_pc + 4.U),
        ((cpu_state === cpu_state_fetch) & latched_store & ~latched_branch) -> Mux(latched_stalu, alu_out_q, reg_out),
      ) ++ (if( ENABLE_IRQ ) { Seq(
        ((cpu_state === cpu_state_fetch) & irq_state.extract(0)) -> (reg_next_pc),
        ((cpu_state === cpu_state_fetch) & irq_state.extract(1)) -> (irq_pending & ~irq_mask),        
      )} else { Seq() })
    )



  val cpuregs_waddr = latched_rd



  val cpuregs_raddr1 = if(ENABLE_REGS_DUALPORT) {decoded_rs1} else { decoded_rs }
  val cpuregs_raddr2 = if(ENABLE_REGS_DUALPORT) {decoded_rs2} else { 0.U(6.W) }

  val cpuregs = Mem(regfile_size, UInt(32.W))

  when( ~reset.asBool & cpuregs_write & latched_rd =/= 0.U ){
    cpuregs(cpuregs_waddr) := cpuregs_wrdata
  }

  val cpuregs_rdata1 = cpuregs(cpuregs_raddr1)
  val cpuregs_rdata2 = cpuregs(cpuregs_raddr2)


  val cpuregs_rs1 = Wire(UInt(32.W))
  val cpuregs_rs2 = Wire(UInt(32.W))


  if (ENABLE_REGS_DUALPORT) {
    cpuregs_rs1 := Mux(decoded_rs1 =/= 0.U, cpuregs_rdata1, 0.U)
    cpuregs_rs2 := Mux(decoded_rs2 =/= 0.U, cpuregs_rdata2, 0.U)
    decoded_rs  := DontCare
  } else {
    decoded_rs  := Mux(cpu_state === cpu_state_ld_rs2, decoded_rs2, decoded_rs1)
    cpuregs_rs1 := Mux(decoded_rs =/= 0.U, cpuregs_rdata1, 0.U)
    cpuregs_rs2 := cpuregs_rs1
  }

  launch_next_insn := (
    if( !ENABLE_IRQ ){
      (cpu_state === cpu_state_fetch) & decoder_trigger
    } else {
      (cpu_state === cpu_state_fetch) & decoder_trigger & ( irq_delay | irq_active | ~(irq_pending & ~irq_mask))
    }    
  )

  

























  reg_out := 0.U

  alu_wait := false.B
  alu_wait_2 := false.B

  when(launch_next_insn){
    dbg_rs1val := 0.U
    dbg_rs2val := 0.U
    dbg_rs1val_valid := false.B
    dbg_rs2val_valid := false.B
  }






























  if( ENABLE_IRQ ){

    val next_irq_pending =
      (
        (irq_pending & LATCHED_IRQ) & (
          Mux( (cpu_state_fetch  === cpu_state & irq_state.extract(1)), irq_mask, "hFFFFFFFF".U(32.W))
        ) |
          Mux( cpu_state_ld_rs1 === cpu_state & instr_trap & ~irq_mask.extract(irq_ebreak) & ~irq_active,
            1.U << irq_ebreak, 0.U
          )
      )

    val busError_irq_pending = Wire( UInt(32.W) )

    if (CATCH_MISALIGN){
      when( ( ~irq_mask.extract(irq_buserror) & ~irq_active ) & (
        (( mem_do_rdata | mem_do_wdata ) & ( ( mem_wordsize === 0.U & reg_op1(1,0) =/= 0.U ) | ( mem_wordsize === 1.U & reg_op1.extract(0) =/= 0.U ) )) |
        (  mem_do_rinst                  & ( reg_pc(1,0) =/= 0.U )                                      )
      ) ){
        busError_irq_pending := (1.U << irq_buserror)
      } .otherwise{
        busError_irq_pending := 0.U
      }  
    } else{
      busError_irq_pending := 0.U
    }




    val irq_pending_dnxt = 
      next_irq_pending | io.irq |
      Mux( (if( ENABLE_IRQ_TIMER ){timer === 1.U} else {false.B}), (1.U << irq_timer), 0.U ) |
      busError_irq_pending

    when(true.B){
      irq_pending := irq_pending_dnxt & ~MASKED_IRQ
    }
  }


  val set_mem_do_wdata = ( cpu_state_stmem === cpu_state ) & (~mem_do_prefetch | mem_done) & ~mem_do_wdata
  val set_mem_do_rdata = ( cpu_state_ldmem === cpu_state ) & (~mem_do_prefetch | mem_done) & ~mem_do_rdata
  val set_mem_do_rinst = WireDefault(false.B)

  when( cpu_state_exec  === cpu_state  ){
    when( if (TWO_CYCLE_ALU || TWO_CYCLE_COMPARE) {(alu_wait | alu_wait_2)} else { false.B }){
    } .elsewhen(is_beq_bne_blt_bge_bltu_bgeu){
      when( if(TWO_CYCLE_COMPARE) {alu_out_0_q} else {alu_out_0}){
        set_mem_do_rinst := true.B
      }      
    }
  }


  if (ENABLE_IRQ & ENABLE_IRQ_TIMER){
    when( timer =/= 0.U ){
      timer := timer - 1.U
    }
  }

  decoder_trigger := mem_do_rinst & mem_done
  decoder_pseudo_trigger := false.B
  do_waitirq := false.B

  trace_valid := false.B




  io.trap := RegNext(cpu_state === cpu_state_trap)


  when( cpu_state_fetch === cpu_state ){
    mem_do_rinst := ~decoder_trigger & ~do_waitirq
    mem_wordsize := 0.U

    val current_pc = 
      Mux(
        latched_branch, Mux(latched_store, Mux(latched_stalu, alu_out_q, reg_out) & "hFFFFFFFE".U, reg_next_pc),
        Mux( (if(ENABLE_IRQ) {irq_state.extract(0)} else {false.B}), PROGADDR_IRQ, reg_next_pc )
      )

    when(latched_branch){
      // printf( "ST_RD:  %d 0x%x, BRANCH 0x%x\n", latched_rd, reg_pc + 4.U, current_pc)
    } .elsewhen(latched_store & ~latched_branch){
      // printf( "ST_RD:  %d 0x%x\n", latched_rd, Mux(latched_stalu, alu_out_q, reg_out))
    }
    
    if( ENABLE_IRQ ){
      when(irq_state.extract(0)){
        irq_active := true.B
        mem_do_rinst := true.B
      } .elsewhen(irq_state.extract(1)){
        eoi := irq_pending & ~irq_mask         
      }      
    }


    if (ENABLE_TRACE){
      when(latched_trace){
        latched_trace := false.B
        trace_valid := true.B
        trace_data :=
          Mux(latched_branch,
            Mux(irq_active, TRACE_IRQ, 0.U) | TRACE_BRANCH | (current_pc & "hfffffffe".U),
            Mux(irq_active, TRACE_IRQ, 0.U) | Mux(latched_stalu, alu_out_q, reg_out)
          ) 
      }
    } else{
      trace_data := 0.U
    }


    reg_pc := current_pc
    reg_next_pc := current_pc

    latched_store  := false.B
    latched_stalu  := false.B
    latched_branch := false.B
    latched_is_lu  := false.B
    latched_is_lh  := false.B
    latched_is_lb  := false.B

    latched_rd := decoded_rd





    if (ENABLE_IRQ){
      when((decoder_trigger & ~irq_active & ~irq_delay & ((irq_pending & ~irq_mask).orR)) | irq_state =/= 0.U ){
        irq_state := Mux(irq_state === "b00".U, "b01".U, Mux(irq_state === "b01".U, "b10".U, "b00".U))
        if (ENABLE_IRQ_QREGS){
          latched_rd := irqregs_offset | irq_state.extract(0)
        } else{
          latched_rd := Mux(irq_state.extract(0), 4.U, 3.U)
        }
      } .elsewhen( (decoder_trigger | do_waitirq) & instr_waitirq ){
        when(irq_pending =/= 0.U){
          latched_store := true.B
          reg_out := irq_pending
          reg_next_pc := current_pc + 4.U
          mem_do_rinst := true.B
        } .otherwise{
          do_waitirq := true.B
        }
      } .elsewhen(decoder_trigger){
        // printf(s"-- %-0t", $time)
        irq_delay := irq_active
        reg_next_pc := current_pc + 4.U
        if (ENABLE_TRACE){
          latched_trace := true.B
        }
        if (ENABLE_COUNTERS){
          count_instr := count_instr + 1.U         
        }
        when(instr_jal){
          mem_do_rinst := true.B
          reg_next_pc := current_pc + decoded_imm_j
          latched_branch := true.B
        } .otherwise{
          mem_do_rinst := false.B
          mem_do_prefetch := ~instr_jalr & ~instr_retirq
        }
      }
    } else {
      when(decoder_trigger){
        // printf(s"-- %-0t", $time)
        irq_delay := irq_active
        reg_next_pc := current_pc + 4.U
        if (ENABLE_TRACE){
          latched_trace := true.B
        }
        if (ENABLE_COUNTERS){
          count_instr := count_instr + 1.U         
        }
        when(instr_jal){
          mem_do_rinst := true.B
          reg_next_pc := current_pc + decoded_imm_j
          latched_branch := true.B
        } .otherwise{
          mem_do_rinst := false.B
          mem_do_prefetch := ~instr_jalr & ~instr_retirq
        }
      }
    }





  } .elsewhen( cpu_state_ld_rs1 === cpu_state ){
    reg_op1 := 0.U
    reg_op2 := 0.U

    when(is_lui_auipc_jal){
      reg_op1 := Mux(instr_lui, 0.U, reg_pc)
      reg_op2 := decoded_imm
      if (TWO_CYCLE_ALU){
        alu_wait := true.B
      } else {
        mem_do_rinst := mem_do_prefetch
      }
    } .elsewhen(is_lb_lh_lw_lbu_lhu & ~instr_trap){
      // printf( "LD_RS1: %d 0x%x\n", decoded_rs1, cpuregs_rs1)
      reg_op1 := cpuregs_rs1
      dbg_rs1val := cpuregs_rs1
      dbg_rs1val_valid := true.B
      mem_do_rinst := true.B
    } .elsewhen( instr_trap ){
      printf( "EBREAK OR UNSUPPORTED INSN AT 0x%x\n", reg_pc)
    } .elsewhen( is_rdcycle_rdcycleh_rdinstr_rdinstrh ){
      when(instr_rdcycle){
        reg_out := count_cycle(31,0)
      }.elsewhen(instr_rdinstr){
        reg_out := count_instr(31,0)
      }

      when(instr_rdinstrh){
        reg_out := count_instr(63,32)
      } .elsewhen(instr_rdcycleh){
        reg_out := count_cycle(63,32)
      } 

      latched_store := true.B
    } .elsewhen( instr_getq ){
      // printf( "LD_RS1: %d 0x%x\n", decoded_rs1, cpuregs_rs1)
      reg_out := cpuregs_rs1
      dbg_rs1val := cpuregs_rs1
      dbg_rs1val_valid := true.B
      latched_store := true.B
    } .elsewhen( instr_setq ){
      // printf( "LD_RS1: %d 0x%x\n", decoded_rs1, cpuregs_rs1)
      reg_out := cpuregs_rs1
      dbg_rs1val := cpuregs_rs1
      dbg_rs1val_valid := true.B
      latched_rd := latched_rd | irqregs_offset
      latched_store := true.B
    } .elsewhen( instr_retirq ){
      eoi := false.B
      irq_active := false.B
      latched_branch := true.B
      latched_store := true.B
      // printf( "LD_RS1: %d 0x%x\n", decoded_rs1, cpuregs_rs1)
      reg_out := (if(CATCH_MISALIGN) {(cpuregs_rs1 & "hfffffffe".U)} else {cpuregs_rs1})
      dbg_rs1val := cpuregs_rs1
      dbg_rs1val_valid := true.B
    } .elsewhen( instr_maskirq ){
      latched_store := true.B
      reg_out := irq_mask
      // printf( "LD_RS1: %d 0x%x\n", decoded_rs1, cpuregs_rs1)
      irq_mask := cpuregs_rs1 | MASKED_IRQ
      dbg_rs1val := cpuregs_rs1
      dbg_rs1val_valid := true.B
    } .elsewhen( instr_timer ){
      latched_store := true.B
      reg_out := timer
      // printf( "LD_RS1: %d 0x%x\n", decoded_rs1, cpuregs_rs1)
      timer := cpuregs_rs1
      dbg_rs1val := cpuregs_rs1
      dbg_rs1val_valid := true.B
    } .elsewhen( is_slli_srli_srai | is_jalr_addi_slti_sltiu_xori_ori_andi ){
      // printf( "LD_RS1: %d 0x%x\n", decoded_rs1, cpuregs_rs1)
      reg_op1 := cpuregs_rs1
      dbg_rs1val := cpuregs_rs1
      dbg_rs1val_valid := true.B
      reg_op2 := Mux(is_slli_srli_srai, decoded_rs2, decoded_imm)
      if (TWO_CYCLE_ALU){
        alu_wait := true.B
      } else {
        mem_do_rinst := mem_do_prefetch
      }
    } .otherwise{
      // printf( "LD_RS1: %d 0x%x\n", decoded_rs1, cpuregs_rs1)
      reg_op1 := cpuregs_rs1
      dbg_rs1val := cpuregs_rs1
      dbg_rs1val_valid := true.B
      if (ENABLE_REGS_DUALPORT){
        // printf( "LD_RS2: %d 0x%x\n", decoded_rs2, cpuregs_rs2)
        reg_op2 := cpuregs_rs2
        dbg_rs2val := cpuregs_rs2
        dbg_rs2val_valid := true.B

        when(is_sb_sh_sw){
          mem_do_rinst := true.B
        } .otherwise{
          when( if(TWO_CYCLE_ALU) {true.B} else {if(TWO_CYCLE_COMPARE) {is_beq_bne_blt_bge_bltu_bgeu} else {false.B}}) {
            alu_wait_2 := (if( TWO_CYCLE_ALU & TWO_CYCLE_COMPARE ) {is_beq_bne_blt_bge_bltu_bgeu} else {false.B})
            alu_wait := true.B
          } .otherwise{
            mem_do_rinst := mem_do_prefetch
          }
        }
      }
    }


  } .elsewhen( cpu_state_ld_rs2 === cpu_state ){
    // printf( "LD_RS2: %d 0x%x\n", decoded_rs2, cpuregs_rs2)
    reg_op2 := cpuregs_rs2
    dbg_rs2val := cpuregs_rs2
    dbg_rs2val_valid := true.B

    when(is_sb_sh_sw){
      mem_do_rinst := true.B
    } .otherwise{
      when( if (TWO_CYCLE_ALU) {true.B} else { if(TWO_CYCLE_COMPARE) {is_beq_bne_blt_bge_bltu_bgeu} else {false.B} } ){
        alu_wait_2 := (if(TWO_CYCLE_ALU & TWO_CYCLE_COMPARE) {is_beq_bne_blt_bge_bltu_bgeu} else {false.B})
        alu_wait := true.B
      } .otherwise{
        mem_do_rinst := mem_do_prefetch
      }
    }


  } .elsewhen( cpu_state_exec  === cpu_state  ){
    reg_out := reg_pc + decoded_imm
    when( if (TWO_CYCLE_ALU || TWO_CYCLE_COMPARE) {(alu_wait | alu_wait_2)} else { false.B }){
      mem_do_rinst := mem_do_prefetch & ~alu_wait_2
      alu_wait := alu_wait_2
    } .elsewhen(is_beq_bne_blt_bge_bltu_bgeu){
      latched_rd := 0.U
      latched_store  := (if(TWO_CYCLE_COMPARE) {alu_out_0_q} else {alu_out_0})
      latched_branch := (if(TWO_CYCLE_COMPARE) {alu_out_0_q} else {alu_out_0})
      when( if(TWO_CYCLE_COMPARE) {alu_out_0_q} else {alu_out_0}){
        decoder_trigger := false.B
      }      
    } .otherwise{
      latched_branch := instr_jalr
      latched_store := true.B
      latched_stalu := true.B
    }
    
  } .elsewhen( cpu_state_stmem === cpu_state ){
    if (ENABLE_TRACE) {
      reg_out := reg_op2
    }

    when(~mem_do_prefetch | mem_done){
      when(~mem_do_wdata){

        mem_wordsize := Mux1H(Seq(
          instr_sb -> 2.U,
          instr_sh -> 1.U,
          instr_sw -> 0.U,            
        ))

        if (ENABLE_TRACE){
          trace_valid := true.B
          trace_data  := Mux(irq_active, TRACE_IRQ, 0.U) | TRACE_ADDR | ((reg_op1 + decoded_imm) & "hffffffff".U)
        }

        reg_op1 := reg_op1 + decoded_imm
      }

      when(~mem_do_prefetch & mem_done){
        decoder_trigger := true.B
        decoder_pseudo_trigger := true.B
      }
    }

  } .elsewhen( cpu_state_ldmem === cpu_state ){
    latched_store := true.B
    when(~mem_do_prefetch | mem_done){
      when(~mem_do_rdata){

          mem_wordsize := Mux1H(Seq(
            (instr_lb | instr_lbu) -> 2.U,
            (instr_lh | instr_lhu) -> 1.U,
            (instr_lw            ) -> 0.U,
          ))

        latched_is_lu := is_lbu_lhu_lw
        latched_is_lh := instr_lh
        latched_is_lb := instr_lb

        if (ENABLE_TRACE){
          trace_valid := true.B
          trace_data := Mux(irq_active, TRACE_IRQ, 0.U) | TRACE_ADDR | ((reg_op1 + decoded_imm) & "hffffffff".U)
        }
        reg_op1 := reg_op1 + decoded_imm      
      }

      when(~mem_do_prefetch & mem_done){

        reg_out := Mux1H(Seq(
          latched_is_lu -> mem_rdata_word,
          latched_is_lh -> Cat( Fill(16, mem_rdata_word.extract(15)), mem_rdata_word(15,0) ),
          latched_is_lb -> Cat( Fill(24, mem_rdata_word.extract(7)) , mem_rdata_word(7,0)  ),
        ))

        decoder_trigger := true.B
        decoder_pseudo_trigger := true.B        
      }
    
    }
  
  }






  if (CATCH_MISALIGN){
    when( (mem_do_rdata | mem_do_wdata)){
      when(mem_wordsize === 0.U & reg_op1(1,0)         =/= 0.U){
        printf( "MISALIGNED WORD: 0x%x\n", reg_op1)       
      }
      when (mem_wordsize === 1.U && reg_op1.extract(0)  =/= 0.U){
        printf( "MISALIGNED HALFWORD: 0x%x\n", reg_op1)
      }
    }
    when( mem_do_rinst & ( reg_pc(1,0) =/= 0.U )){
      printf( "MISALIGNED INSTRUCTION: 0x%x\n", reg_pc)
    }
  }
















































  when( cpu_state_fetch === cpu_state ){
    when( decoder_trigger ){
      when(( ~irq_active & ~irq_delay & ((irq_pending & ~irq_mask).orR)) | irq_state =/= 0.U ){

      } .elsewhen( instr_waitirq ){

      } .elsewhen(~instr_jal){
        cpu_state := cpu_state_ld_rs1
      }      
    }

  } .elsewhen( cpu_state_ld_rs1 === cpu_state ){
    when(is_lui_auipc_jal | is_slli_srli_srai | is_jalr_addi_slti_sltiu_xori_ori_andi){
      cpu_state := cpu_state_exec
    } .elsewhen(is_lb_lh_lw_lbu_lhu & ~instr_trap){
      cpu_state := cpu_state_ldmem
    } .elsewhen( instr_trap ){
      when( ~irq_mask.extract(irq_ebreak) & ~irq_active ){
        cpu_state := cpu_state_fetch
      } .otherwise{
        cpu_state := cpu_state_trap
      }
    } .elsewhen( is_rdcycle_rdcycleh_rdinstr_rdinstrh | instr_getq | instr_setq | instr_retirq | instr_maskirq | instr_timer ){
      cpu_state := cpu_state_fetch
    } .otherwise{
      if (ENABLE_REGS_DUALPORT){
        when(is_sb_sh_sw){
          cpu_state := cpu_state_stmem
        } .otherwise{
          cpu_state := cpu_state_exec
        }
      } else {
        cpu_state := cpu_state_ld_rs2
      }
    }

  } .elsewhen( cpu_state_ld_rs2 === cpu_state ){ // only ENABLE_REGS_DUALPORT will entry this state
    when(is_sb_sh_sw){
      cpu_state := cpu_state_stmem
    } .otherwise{
      cpu_state := cpu_state_exec
    }
  } .elsewhen( cpu_state_exec === cpu_state ){
    when( if (TWO_CYCLE_ALU || TWO_CYCLE_COMPARE) {(alu_wait | alu_wait_2)} else { false.B }){
    } .elsewhen(is_beq_bne_blt_bge_bltu_bgeu){
      when(mem_done){
        cpu_state := cpu_state_fetch
      }
    } .otherwise{
      cpu_state := cpu_state_fetch
    }
    
  } .elsewhen( cpu_state_stmem === cpu_state ){
    when(~mem_do_prefetch & mem_done){
      cpu_state := cpu_state_fetch
    }
  } .elsewhen( cpu_state_ldmem === cpu_state ){
    when(~mem_do_prefetch & mem_done){
      cpu_state := cpu_state_fetch
    }
  }

  if(CATCH_MISALIGN){
    when(~( ~irq_mask.extract(irq_buserror) & ~irq_active)){
      when( (mem_do_rdata | mem_do_wdata)){
        when( (mem_wordsize === 1.U & reg_op1.extract(0) =/= 0.U) | (mem_wordsize === 0.U & reg_op1(1,0) =/= 0.U) ){
          cpu_state := cpu_state_trap            
        }
      }
      when( mem_do_rinst & ( reg_pc(1,0) =/= 0.U ) ){
        cpu_state := cpu_state_trap
      }        
    }
  } else{
    when( decoder_trigger_q & ~decoder_pseudo_trigger_q & instr_ecall_ebreak){
      cpu_state := cpu_state_trap
    }    
  }

































  when(mem_done){
    mem_do_prefetch := false.B
    mem_do_rinst := false.B
    mem_do_rdata := false.B
    mem_do_wdata := false.B      
  }

  when(set_mem_do_rinst){
    mem_do_rinst := true.B      
  }

  when(set_mem_do_rdata){
    mem_do_rdata := true.B      
  }

  when(set_mem_do_wdata){
    mem_do_wdata := true.B      
  }




  if (!CATCH_MISALIGN){
    reg_pc(1,0) := 0.U
    reg_next_pc(1,0) := 0.U
  }







  if(RISCV_FORMAL){
    val dbg_irq_call  = RegInit(false.B)
    val dbg_irq_enter = RegInit(false.B)
    val dbg_irq_ret   = Reg(UInt(32.W))

    io.rvfi.get.valid := RegNext( (launch_next_insn | io.trap) & dbg_valid_insn, false.B)
    io.rvfi.get.order := RegNext( io.rvfi.get.order + io.rvfi.get.valid, 0.U)

    io.rvfi.get.insn      := RegNext(dbg_insn_opcode)
    io.rvfi.get.rs2_addr  := RegNext(Mux(dbg_rs2val_valid, dbg_insn_rs2, 0.U))
    io.rvfi.get.pc_rdata  := RegNext(dbg_insn_addr)
    io.rvfi.get.rs2_rdata := RegNext(Mux(dbg_rs2val_valid, dbg_rs2val, 0.U))
    io.rvfi.get.trap      := RegNext(io.trap)
    io.rvfi.get.halt      := RegNext(io.trap)
    io.rvfi.get.intr      := RegNext(dbg_irq_enter)
    io.rvfi.get.mode      := RegNext(3.U)
    io.rvfi.get.ixl       := RegNext(1.U)



    when(io.rvfi.get.valid){
      dbg_irq_call := false.B
      dbg_irq_enter := dbg_irq_call
    } .elsewhen(irq_state === 1.U){
      dbg_irq_call := true.B
      dbg_irq_ret := next_pc
    }

    val rvfi_rd_addr  = RegInit(0.U(5.W));  io.rvfi.get.rd_addr  := rvfi_rd_addr
    val rvfi_rd_wdata = RegInit(0.U(32.W)); io.rvfi.get.rd_wdata := rvfi_rd_wdata

    when( dbg_insn_opcode === BitPat("b0000001?????????????000??0001011")){ // setq
      rvfi_rd_addr  := 0.U
      rvfi_rd_wdata := 0.U
    } .elsewhen(cpuregs_write & irq_state === 0.U){
      rvfi_rd_addr  := latched_rd
      rvfi_rd_wdata := Mux(latched_rd =/= 0.U, cpuregs_wrdata, 0.U)      
    } .elsewhen(io.rvfi.get.valid){
      rvfi_rd_addr  := 0.U
      rvfi_rd_wdata := 0.U
    }




    io.rvfi.get.rs1_addr  := RegNext( Mux(dbg_insn_opcode === BitPat("b0000000?????000??????????0001011") | dbg_insn_opcode === BitPat("b0000010?????00000???000000001011"), 0.U, Mux(dbg_rs1val_valid, dbg_insn_rs1, 0.U)))
    io.rvfi.get.rs1_rdata := RegNext( Mux(dbg_insn_opcode === BitPat("b0000000?????000??????????0001011") | dbg_insn_opcode === BitPat("b0000010?????00000???000000001011"), 0.U, Mux(dbg_rs1val_valid, dbg_rs1val,   0.U)))

    val rvfi_mem_addr  = Reg(UInt(32.W)); io.rvfi.get.mem_addr  := rvfi_mem_addr
    val rvfi_mem_rmask = Reg(UInt(4.W)) ; io.rvfi.get.mem_rmask := rvfi_mem_rmask
    val rvfi_mem_wmask = Reg(UInt(4.W)) ; io.rvfi.get.mem_wmask := rvfi_mem_wmask
    val rvfi_mem_rdata = Reg(UInt(32.W)); io.rvfi.get.mem_rdata := rvfi_mem_rdata
    val rvfi_mem_wdata = Reg(UInt(32.W)); io.rvfi.get.mem_wdata := rvfi_mem_wdata

    when(~dbg_irq_call){
      when(dbg_mem_instr){
        rvfi_mem_addr  := 0.U
        rvfi_mem_rmask := 0.U
        rvfi_mem_wmask := 0.U
        rvfi_mem_rdata := 0.U
        rvfi_mem_wdata := 0.U
      } .elsewhen(dbg_mem_valid & dbg_mem_ready){
        rvfi_mem_addr  := dbg_mem_addr
        rvfi_mem_rmask := Mux(dbg_mem_wstrb =/= 0.U, 0.U, "hF".U)
        rvfi_mem_wmask := dbg_mem_wstrb
        rvfi_mem_rdata := dbg_mem_rdata
        rvfi_mem_wdata := dbg_mem_wdata
      }
    }
  




    io.rvfi.get.pc_wdata := Mux(dbg_irq_call, dbg_irq_ret, dbg_insn_addr)

    io.rvfi.get.csr_mcycle_rmask := 0.U
    io.rvfi.get.csr_mcycle_wmask := 0.U
    io.rvfi.get.csr_mcycle_rdata := 0.U
    io.rvfi.get.csr_mcycle_wdata := 0.U

    io.rvfi.get.csr_minstret_rmask := 0.U
    io.rvfi.get.csr_minstret_wmask := 0.U
    io.rvfi.get.csr_minstret_rdata := 0.U
    io.rvfi.get.csr_minstret_wdata := 0.U

    when(io.rvfi.get.valid){
      when( io.rvfi.get.insn === BitPat("b110000000000?????010?????1110011")){
        io.rvfi.get.csr_mcycle_rmask := "h00000000FFFFFFFF".U
        io.rvfi.get.csr_mcycle_rdata := rvfi_rd_wdata
      }
      when(io.rvfi.get.insn === BitPat("b110010000000?????010?????1110011")){
        io.rvfi.get.csr_mcycle_rmask := "hFFFFFFFF00000000".U
        io.rvfi.get.csr_mcycle_rdata := Cat(io.rvfi.get.rd_wdata, 0.U(32.W))
      }
      when(io.rvfi.get.insn === BitPat("b110000000010?????010?????1110011")){
        io.rvfi.get.csr_minstret_rmask :="h00000000FFFFFFFF".U
        io.rvfi.get.csr_minstret_rdata := io.rvfi.get.rd_wdata
      }
      when(io.rvfi.get.insn === BitPat("b110010000010?????010?????1110011")){
        io.rvfi.get.csr_minstret_rmask := "hFFFFFFFF00000000".U
        io.rvfi.get.csr_minstret_rdata := Cat(io.rvfi.get.rd_wdata, 0.U(32.W))
      }      
    }
  }



}


