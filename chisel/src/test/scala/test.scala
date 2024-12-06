
package Picorv32

import chisel3._
import chisel3.stage._

object testMain extends App {
   (new chisel3.stage.ChiselStage).execute( Array("--target-dir", "generated/", "-E", "verilog" ) ++ args, Seq(
      ChiselGeneratorAnnotation(() => { new Picorv32 })
    ))   

  // circt.stage.ChiselStage.emitSystemVerilogFile(
  //   new Picorv32, args = Array( "--target-dir", "generated/"),
  //   firtoolOpts = Array(
  //   "-disable-all-randomization",
  //   "-strip-debug-info",
  //   "--verilog")
  // )
}
