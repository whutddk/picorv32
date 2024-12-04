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



class Pcpi_fast_mul(
    EXTRA_MUL_FFS: Boolean = false,
    EXTRA_INSN_FFS: Boolean = false,
  ) extends Module{
  val io: PCPI_Access_Bundle = IO(new PCPI_Access_Bundle)


	val pcpi_insn_valid = io.valid & io.insn(6,0) == "b0110011".U & io.insn(31,25) == "b0000001".U
	val pcpi_insn_valid_q = RegNext(pcpi_insn_valid)

  val instr_mul    = (if(EXTRA_INSN_FFS) {pcpi_insn_valid_q} else {pcpi_insn_valid}) & io.insn(14,12) === "b000".U
  val instr_mulh   = (if(EXTRA_INSN_FFS) {pcpi_insn_valid_q} else {pcpi_insn_valid}) & io.insn(14,12) === "b001".U
  val instr_mulhsu = (if(EXTRA_INSN_FFS) {pcpi_insn_valid_q} else {pcpi_insn_valid}) & io.insn(14,12) === "b010".U
  val instr_mulhu  = (if(EXTRA_INSN_FFS) {pcpi_insn_valid_q} else {pcpi_insn_valid}) & io.insn(14,12) === "b011".U


	val instr_any_mul    = instr_mul | instr_mulh | instr_mulhsu | instr_mulhu
	val instr_any_mulh   = instr_mulh | instr_mulhsu | instr_mulhu
	val instr_rs1_signed = instr_mulh | instr_mulhsu
	val instr_rs2_signed = instr_mulh

	val shift_out = RegNext(instr_any_mulh)
  val active = RegInit(0.U(4.W))

  val rs1 = Reg(UInt(32.W))
  val rs2 = Reg(UInt(32.W))
  val rs1_q = RegEnable(rs1, active.extract(0))
  val rs2_q = RegEnable(rs2, active.extract(0))

  val rd = RegEnable( (if(EXTRA_MUL_FFS){ rs1_q.asSInt * rs2_q.asSInt  } else { rs1.asSInt * rs2.asSInt })  , active.extract(1))
  val rd_q = RegEnable( rd, active.extract(2))




    


		when( instr_any_mul & (if( EXTRA_MUL_FFS ){ active === 0.U } else{ active(1,0) === 0.U }) ){
			rs1 := Mux( instr_rs1_signed, io.rs1.asSInt, io.rs1 )
			rs2 := Mux( instr_rs2_signed, io.rs2.asSInt, io.rs2 )
      active := Cat( active(2,0), 1.U(1.W) )
    } .otherwise{
			active := Cat( active(2,0), 0.U(1.W) )
    }


	io.wr    := (if( EXTRA_MUL_FFS ) {active.extract(3)} else {active.extract(1)})
	io.ready := (if( EXTRA_MUL_FFS ) {active.extract(3)} else {active.extract(1)})
	io.wait  := false.B


  if(RISCV_FORMAL_ALTOPS){
    io.rd := Mux1H(Seq(
      instr_mul    -> ( (io.rs1 + io.rs2) ^ "h5876063e".U ),
      instr_mulh   -> ( (io.rs1 + io.rs2) ^ "hf6583fb7".U ),
      instr_mulhsu -> ( (io.rs1 - io.rs2) ^ "hecfbe137".U ),
      instr_mulhu  -> ( (io.rs1 + io.rs2) ^ "h949ce5e8".U ),
    ))
  } else{
    io.rd := (
      if( EXTRA_MUL_FFS ){
        Mux( shift_out, rd_q >> 32, rd_q )
      } else{
        Mux( shift_out, rd >> 32, rd )
      }      
    )

  }



}

