
package Picorv32



object testMain extends App {
  circt.stage.ChiselStage.emitSystemVerilogFile(
    new Picorv32,
    firtoolOpts = Array(
    "--target-dir", "generated/",
    "-disable-all-randomization",
    "-strip-debug-info",
    "--disallowLocalVariables",
    "--disallowPackedArrays")
  )
}
