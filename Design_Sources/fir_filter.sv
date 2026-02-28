// Module: fir_filter
// Description: 128-tap FIR lowpass filter
//              Cutoff: 8 kHz, Fs: 200 kSPS, Window: Hamming
//              Q1.15 fixed-point coefficients (sum=32764, ~unity DC gain)
//              Time-multiplexed MAC: 128 cycles to compute one output
//              At 200kSPS input rate there are 500 clock cycles per sample,
//              so 128 cycles per output leaves comfortable margin.
//
//              Pipeline: 2-stage multiply to meet timing
//                Stage 1: register coeff + data inputs
//                Stage 2: registered multiply output
//              Extra OUTPUT state flushes the pipeline after last tap
//
//              Output width 28 bits: 12 (data) + 16 (coeff) = 28 product bits
//              Normalise by >>> 15 to remove Q1.15 scaling

module fir_filter #(
    parameter int COEFF_WIDTH  = 16,   // Q1.15
    parameter int DATA_WIDTH   = 12,   // XADC output
    parameter int OUTPUT_WIDTH = 28    // 12 + 16 = 28, enough to prevent overflow
)(
    input  logic                              clk,
    input  logic                              reset,
    input  logic signed [DATA_WIDTH-1:0]      data_in,
    input  logic                              data_valid,
    output logic signed [OUTPUT_WIDTH-1:0]    data_out,
    output logic                              data_out_valid
);

    // Q1.15 FIR coefficients
    // Fc=8kHz, Fs=200kHz, 128 taps, Hamming window
    localparam signed [15:0] COEFFS [0:127] = '{
        -16'sd3, 16'sd0, 16'sd3, 16'sd7, 16'sd11, 16'sd14, 16'sd17, 16'sd20,
        16'sd21, 16'sd22, 16'sd20, 16'sd17, 16'sd12, 16'sd4, -16'sd5, -16'sd16,
        -16'sd28, -16'sd40, -16'sd52, -16'sd62, -16'sd69, -16'sd71, -16'sd69, -16'sd61,
        -16'sd47, -16'sd26, 16'sd0, 16'sd31, 16'sd64, 16'sd99, 16'sd131, 16'sd159,
        16'sd180, 16'sd191, 16'sd189, 16'sd173, 16'sd141, 16'sd95, 16'sd35, -16'sd37,
        -16'sd118, -16'sd202, -16'sd285, -16'sd360, -16'sd420, -16'sd461, -16'sd474, -16'sd455,
        -16'sd400, -16'sd307, -16'sd173, 16'sd0, 16'sd209, 16'sd448, 16'sd712, 16'sd992,
        16'sd1278, 16'sd1560, 16'sd1827, 16'sd2069, 16'sd2275, 16'sd2438, 16'sd2551, 16'sd2608,
        16'sd2608, 16'sd2551, 16'sd2438, 16'sd2275, 16'sd2069, 16'sd1827, 16'sd1560, 16'sd1278,
        16'sd992, 16'sd712, 16'sd448, 16'sd209, 16'sd0, -16'sd173, -16'sd307, -16'sd400,
        -16'sd455, -16'sd474, -16'sd461, -16'sd420, -16'sd360, -16'sd285, -16'sd202, -16'sd118,
        -16'sd37, 16'sd35, 16'sd95, 16'sd141, 16'sd173, 16'sd189, 16'sd191, 16'sd180,
        16'sd159, 16'sd131, 16'sd99, 16'sd64, 16'sd31, 16'sd0, -16'sd26, -16'sd47,
        -16'sd61, -16'sd69, -16'sd71, -16'sd69, -16'sd62, -16'sd52, -16'sd40, -16'sd28,
        -16'sd16, -16'sd5, 16'sd4, 16'sd12, 16'sd17, 16'sd20, 16'sd22, 16'sd21,
        16'sd20, 16'sd17, 16'sd14, 16'sd11, 16'sd7, 16'sd3, 16'sd0, -16'sd3
    };

    // Delay line
    logic signed [DATA_WIDTH-1:0] shift_reg [0:127];

    // State machine
    typedef enum logic [1:0] {
        IDLE,
        COMPUTE,
        OUTPUT
    } state_t;

    state_t state;
    logic [6:0] tap_index;

    // Accumulator - needs to hold sum of 128 products
    // Each product is DATA_WIDTH + COEFF_WIDTH = 28 bits
    // 128 products -> need 7 more bits = 35 bits total, use 36 for safety
    logic signed [35:0] accumulator;

    // 2-stage multiply pipeline
    logic signed [COEFF_WIDTH-1:0]              coeff_reg;
    logic signed [DATA_WIDTH-1:0]               data_reg;
    logic signed [COEFF_WIDTH+DATA_WIDTH-1:0]   product_p1;   // combinational multiply
    logic signed [COEFF_WIDTH+DATA_WIDTH-1:0]   product_p2;   // registered output

    // Pipeline delay tracking
    logic compute_active;
    logic compute_active_d1;

    //==========================================================================
    // Stage 1: Register inputs to multiplier
    //==========================================================================
    always_ff @(posedge clk) begin
        if (reset) begin
            coeff_reg <= '0;
            data_reg  <= '0;
        end else if (state == COMPUTE) begin
            coeff_reg <= COEFFS[tap_index];
            data_reg  <= shift_reg[tap_index];
        end
    end

    //==========================================================================
    // Stage 2: Combinational multiply (registered inputs prevent long path)
    //==========================================================================
    always_comb begin
        product_p1 = coeff_reg * data_reg;
    end

    //==========================================================================
    // Stage 3: Register multiply output
    //==========================================================================
    always_ff @(posedge clk) begin
        if (reset)
            product_p2 <= '0;
        else
            product_p2 <= product_p1;
    end

    //==========================================================================
    // Pipeline control
    //==========================================================================
    always_ff @(posedge clk) begin
        if (reset) begin
            compute_active    <= 1'b0;
            compute_active_d1 <= 1'b0;
        end else begin
            compute_active    <= (state == COMPUTE);
            compute_active_d1 <= compute_active;
        end
    end

    //==========================================================================
    // State machine + MAC
    //==========================================================================
    always_ff @(posedge clk) begin
        if (reset) begin
            state          <= IDLE;
            tap_index      <= '0;
            accumulator    <= '0;
            data_out       <= '0;
            data_out_valid <= 1'b0;
            for (int i = 0; i < 128; i++)
                shift_reg[i] <= '0;
        end else begin
            data_out_valid <= 1'b0;

            case (state)
                IDLE: begin
                    if (data_valid) begin
                        // Shift in new sample
                        shift_reg[0] <= data_in;
                        for (int i = 1; i < 128; i++)
                            shift_reg[i] <= shift_reg[i-1];
                        state       <= COMPUTE;
                        tap_index   <= 0;
                        accumulator <= '0;
                    end
                end

                COMPUTE: begin
                    // Accumulate pipeline result (2 cycles delayed)
                    if (compute_active_d1)
                        accumulator <= accumulator + product_p2;

                    if (tap_index == 127)
                        state <= OUTPUT;
                    else
                        tap_index <= tap_index + 1;
                end

                OUTPUT: begin
                    // Wait for pipeline to flush then capture final result
                    if (!compute_active_d1) begin
                        data_out       <= (accumulator + product_p2) >>> 15;
                        data_out_valid <= 1'b1;
                        state          <= IDLE;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule