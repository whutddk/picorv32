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

package Picorv32

import chisel3._
import chisel3.util._



/***************************************************************
 * picorv32_pcpi_div
 ***************************************************************/

class Pcpi_div extends Module{
  val io: PCPI_Access_Bundle = IO( Flipped(new PCPI_Access_Bundle) )

	val pcpi_wr  = Reg(Bool());    io.wr := pcpi_wr
	val pcpi_rd = Reg(UInt(32.W)); io.rd := pcpi_rd
	val pcpi_ready = Reg(Bool());  io.ready := pcpi_ready


  val instr_div  = RegNext( io.valid & ~io.ready & io.insn(6,0) === "b0110011".U & io.insn(31,25) === "b0000001".U & io.insn(14,12) === "b100".U, false.B)
  val instr_divu = RegNext( io.valid & ~io.ready & io.insn(6,0) === "b0110011".U & io.insn(31,25) === "b0000001".U & io.insn(14,12) === "b101".U, false.B)
  val instr_rem  = RegNext( io.valid & ~io.ready & io.insn(6,0) === "b0110011".U & io.insn(31,25) === "b0000001".U & io.insn(14,12) === "b110".U, false.B)
  val instr_remu = RegNext( io.valid & ~io.ready & io.insn(6,0) === "b0110011".U & io.insn(31,25) === "b0000001".U & io.insn(14,12) === "b111".U, false.B)

  val instr_any_div_rem = instr_div | instr_divu | instr_rem | instr_remu
  val pcpi_waiting  = RegNext( instr_any_div_rem, false.B);  io.waiting := pcpi_waiting

  val pcpi_waiting_q = RegNext(pcpi_waiting, false.B)
  val start = pcpi_waiting & ~pcpi_waiting_q






	val dividend = Reg(UInt(32.W))
	val divisor = Reg(UInt(63.W))
	val quotient = Reg(UInt(32.W))
	val quotient_msk = Reg(UInt(32.W))
	val running = RegInit(false.B)
	val outsign = Reg(Bool())



		when(start) {
			running := true.B
			dividend := Mux( (instr_div | instr_rem) & io.rs1.extract(31), -io.rs1, io.rs1)
			divisor  := Mux( (instr_div | instr_rem) & io.rs2.extract(31), -io.rs2, io.rs2) << 31
			outsign  := (instr_div & (io.rs1.extract(31) ^ io.rs2.extract(31)) & io.rs2.orR) | (instr_rem & io.rs1.extract(31))
			quotient := false.B
			quotient_msk := 1.U << 31
      pcpi_ready := false.B
      pcpi_wr := false.B
      pcpi_rd := 0.U
    } .elsewhen( quotient_msk === 0.U & running){
			running := false.B
			pcpi_ready := true.B
			pcpi_wr := true.B
      

      pcpi_rd := Mux( instr_div | instr_divu, Mux(outsign, -quotient, quotient), Mux(outsign, -dividend, dividend) )     

   
    } .otherwise{
			when(divisor <= dividend){
				dividend := dividend - divisor
				quotient := quotient | quotient_msk
      }
      pcpi_ready := false.B
      pcpi_wr := false.B
      pcpi_rd := 0.U
			divisor := divisor >> 1


      quotient_msk := quotient_msk >> 1                      

      
    }














}

