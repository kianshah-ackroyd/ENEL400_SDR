// Module: am_demodulator
// Description: AM envelope detector using Xilinx CORDIC IP in Translate mode
//              Computes magnitude = sqrt(I^2 + Q^2)
//              This is the AM envelope - the instantaneous signal amplitude
//
// CORDIC IP Settings:
//   Functional Selection:      Translate
//   Architectural Config:      Parallel
//   Pipelining Mode:           Maximum
//   Input Width:               16 bits (sign-extend our 13-bit data)
//   Output Width:              16 bits
//   Round Mode:                Truncate
//   Compensation Scaling:      No Scale Compensation
//   Flow Control:              Blocking (default)
//
// CORDIC TDATA packing (Translate mode):
//   s_axis_cartesian_tdata = {Y[15:0], X[15:0]}  (32 bits total)
//   X = I channel (sign extended to 16 bits)
//   Y = Q channel (sign extended to 16 bits)
//
// Output: magnitude in bits [11:0] of m_axis_dout_tdata
//         (upper bits contain phase angle which we discard)
//
// CORDIC gain: ~1.647x (not compensated - only relative amplitude matters for AM)

module am_demodulator #(
    parameter int INPUT_WIDTH  = 13,
    parameter int OUTPUT_WIDTH = 13
)(
    input  logic                              clk,
    input  logic                              reset,

    input  logic signed [INPUT_WIDTH-1:0]     i_data,
    input  logic signed [INPUT_WIDTH-1:0]     q_data,
    input  logic                              data_valid,

    output logic [OUTPUT_WIDTH-1:0]           magnitude,
    output logic                              magnitude_valid
);

    // Sign-extend inputs to 16 bits for CORDIC
    logic signed [15:0] i_extended;
    logic signed [15:0] q_extended;

    always_comb begin
        i_extended = {{(16-INPUT_WIDTH){i_data[INPUT_WIDTH-1]}}, i_data};
        q_extended = {{(16-INPUT_WIDTH){q_data[INPUT_WIDTH-1]}}, q_data};
    end

    // CORDIC output - Translate mode gives {phase[15:0], magnitude[15:0]}
    logic [31:0] cordic_dout;
    logic        cordic_valid;

    // Xilinx CORDIC IP instance
    cordic_0 cordic_inst (
        .aclk                        (clk),

        // Input: packed {Y[15:0], X[15:0]} = {Q, I}
        .s_axis_cartesian_tdata      ({q_extended, i_extended}),
        .s_axis_cartesian_tvalid     (data_valid),

        // Output: packed {phase[15:0], magnitude[15:0]}
        .m_axis_dout_tdata           (cordic_dout),
        .m_axis_dout_tvalid          (cordic_valid)
    );

    // Register output - take lower OUTPUT_WIDTH bits of magnitude word
    // magnitude is in cordic_dout[15:0], phase is in cordic_dout[31:16]
    always_ff @(posedge clk) begin
        if (reset) begin
            magnitude       <= '0;
            magnitude_valid <= 1'b0;
        end else begin
            magnitude_valid <= cordic_valid;
            if (cordic_valid)
                magnitude <= cordic_dout[OUTPUT_WIDTH-1:0];
        end
    end

endmodule
