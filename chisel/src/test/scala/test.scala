
package Picorv32

import chisel3._
import chisel3.util._

object testModule extends App {
  (new chisel3.stage.ChiselStage).execute( Array("--target-dir", "generated/", "-e", "verilog" ) ++ args, Seq(
      ChiselGeneratorAnnotation(() => { new Picorv32 })
    ))
}
