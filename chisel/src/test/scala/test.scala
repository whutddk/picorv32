
package Picorv32



object testMain extends App {
  circt.stage.ChiselStage.emitSystemVerilogFile(
    new Picorv32, args = Array( "--target-dir", "generated/"),
    firtoolOpts = Array(
    "-disable-all-randomization",
    "-strip-debug-info")
  )
}
