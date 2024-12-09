module mem(
    input clock,
    input         mem_valid,
    output reg    mem_ready,

    input  [31:0] mem_addr,
    input  [31:0] mem_wdata,
    input  [3:0]  mem_wstrb,
    output reg [31:0] mem_rdata,
    output reg tests_passed
);

    reg [31:0] sram[0:128*1024/4-1];

    always @(posedge clock) begin
        if( mem_valid ) begin
            if( mem_wstrb == 4'b0 )begin //read
                mem_rdata <= sram[ mem_addr[15:0] ];
            end else begin
                if(mem_wstrb[3]) begin
                    sram[ mem_addr[15:0] ][31:24] <= mem_wdata[31:24];
                end
                if(mem_wstrb[2]) begin
                    sram[ mem_addr[15:0] ][23:16] <= mem_wdata[23:16];
                end
                if(mem_wstrb[1]) begin
                    sram[ mem_addr[15:0] ][15: 8] <= mem_wdata[15: 8];
                end
                if(mem_wstrb[0]) begin
                    sram[ mem_addr[15:0] ][ 7: 0] <= mem_wdata[ 7: 0];
                end
            end
        end

        if( mem_valid & ~mem_ready ) begin
            mem_ready <= 1'b1;
        end else begin
            mem_ready <= 1'b0;
        end
    end

    initial begin
        tests_passed = 0;
    end

    always @(posedge clock) begin
        if ( mem_valid & mem_ready & mem_wstrb != 4'b0 ) begin

            if ( mem_addr == 32'h20000000 & mem_wdata == 123456789 ) begin
                tests_passed = 1;
            end
            if( mem_addr == 32'h10000000 ) begin
                $write("%c", mem_wdata[7:0]);
            end
        end
    end

endmodule
