`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 02.02.2026 21:46:35
// Design Name: 
// Module Name: wave_660
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module wave
    #(parameter int WIDTH = 32,
      parameter int CLK_FREQ = 100_000_000,
      parameter logic [WIDTH*2-1:0] SIGNAL = 660_000
          
    )(
    input logic clk,
    input logic reset,
    output logic [1:0] cycle
    );    
    
    logic [WIDTH-1:0] adder;
    logic [WIDTH-1:0] square;
    
    
    assign adder = (SIGNAL << WIDTH)/CLK_FREQ;
    
    always_ff @(posedge clk) begin
        if (reset) begin
            square <= 0;
            
        end else begin      
            square <= square + adder; 
            
        end
    end
    
    assign cycle = square[WIDTH-1:WIDTH-2];
    
endmodule
