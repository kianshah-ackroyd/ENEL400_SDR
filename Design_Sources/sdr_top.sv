
//              Full chain: XADC -> DC remove -> FIR -> CORDIC -> DC block -> PWM
//
//              IIR DC blocker between demodulator and PWM:
//              y[n] = x[n] - x[n-1] + 0.995 * y[n-1]
//              This removes the constant magnitude floor (DC offset from XADC
//              floating inputs and CORDIC gain) while passing audio frequencies.
//              Cutoff ~16Hz at 10kSPS demodulator output rate.


module sdr_top(
    input  logic        clk,
    input  logic        reset,

    input  logic        Vauxp14,
    input  logic        Vauxn14,
    input  logic        Vauxp6,
    input  logic        Vauxn6,

    output logic [1:0]  cycle,
    output logic        pwm_audio_out,
    output logic [7:0]  led
);

    // Power-On Reset
    logic [23:0] por_counter;
    logic        reset_internal;

    always_ff @(posedge clk) begin
        if (por_counter < 24'd10_000_000)
            por_counter <= por_counter + 1;
    end
    
    assign reset_internal = (por_counter < 24'd10_000_000) | reset;


    // Local Oscillator
    wave #(
        .WIDTH(32),
        .CLK_FREQ(100_000_000),
        .SIGNAL(660_000)
    ) local_oscillator (
        .clk(clk),
        .reset(reset_internal),
        .cycle(cycle)
    );


    // XADC
    logic [11:0] adc_i_data;
    logic [11:0] adc_q_data;
    logic        adc_data_valid;

    xadc_reader #(
        .DATA_WIDTH(12)
    ) xadc_inst (
        .clk        (clk),
        .reset      (reset_internal),
        .Vauxp14    (Vauxp14),
        .Vauxn14    (Vauxn14),
        .Vauxp6     (Vauxp6),
        .Vauxn6     (Vauxn6),
        .i_data     (adc_i_data),
        .q_data     (adc_q_data),
        .data_valid (adc_data_valid),
        .eoc        (),
        .eos        ()
    );


    // DC Removal (XADC offset)
    logic signed [12:0] i_centered;
    logic signed [12:0] q_centered;
    // XADC centers reading in the middle of the voltage range (0.5*2^12 = 2048). 
    // Subtracting by 2048 centers reading at 0.
    always_comb begin
        i_centered = $signed({1'b0, adc_i_data}) - 13'sd2048;
        q_centered = $signed({1'b0, adc_q_data}) - 13'sd2048;
    end


    // FIR Filter - I and Q
    logic signed [27:0] fir_i_out;
    logic signed [27:0] fir_q_out;
    logic               fir_i_valid;
    logic               fir_q_valid;

    fir_filter #(
        .COEFF_WIDTH (16),
        .DATA_WIDTH  (12),
        .OUTPUT_WIDTH(28)
    ) fir_i (
        .clk           (clk),
        .reset         (reset_internal),
        .data_in       (i_centered[11:0]),
        .data_valid    (adc_data_valid),
        .data_out      (fir_i_out),
        .data_out_valid(fir_i_valid)
    );

    fir_filter #(
        .COEFF_WIDTH (16),
        .DATA_WIDTH  (12),
        .OUTPUT_WIDTH(28)
    ) fir_q (
        .clk           (clk),
        .reset         (reset_internal),
        .data_in       (q_centered[11:0]),
        .data_valid    (adc_data_valid),
        .data_out      (fir_q_out),
        .data_out_valid(fir_q_valid)
    );


    // Scale FIR output to 13 bits for CORDIC
    logic signed [12:0] fir_i_scaled;
    logic signed [12:0] fir_q_scaled;

    always_comb begin
        fir_i_scaled = fir_i_out[12:0];
        fir_q_scaled = fir_q_out[12:0];
    end


    // AM Demodulator (CORDIC magnitude)
    logic [12:0] magnitude;
    logic        magnitude_valid;

    am_demodulator #(
        .INPUT_WIDTH (13),
        .OUTPUT_WIDTH(13)
    ) demodulator (
        .clk            (clk),
        .reset          (reset_internal),
        .i_data         (fir_i_scaled),
        .q_data         (fir_q_scaled),
        .data_valid     (fir_i_valid),
        .magnitude      (magnitude),
        .magnitude_valid(magnitude_valid)
    );


    // IIR DC Blocker
    // Removes the constant magnitude floor from the demodulated signal
    // y[n] = x[n] - x[n-1] + alpha * y[n-1]
    // alpha = 0.995 implemented as: y - (y >>> 7) i.e. 1 - 1/128 ~ 0.992
    // Cutoff ~16Hz at 10kSPS - well below voice frequencies
    //
    // Input:  13-bit unsigned magnitude (0 to 8191)
    // Output: signed, centred around 0, message rides above/below baseline
    
    logic signed [20:0] dc_block_y;      // Extra bits for IIR accumulation
    logic signed [20:0] dc_block_x_prev;
    logic signed [12:0] dc_blocked_out;
    logic               dc_blocked_valid;

    always_ff @(posedge clk) begin
        if (reset_internal) begin
            dc_block_y     <= '0;
            dc_block_x_prev <= '0;
            dc_blocked_out  <= '0;
            dc_blocked_valid <= 1'b0;
        end else begin
            dc_blocked_valid <= 1'b0;
            if (magnitude_valid) begin
                // y[n] = x[n] - x[n-1] + y[n-1] - (y[n-1] >>> 7)
                dc_block_y      <= $signed({1'b0, magnitude}) 
                                   - dc_block_x_prev 
                                   + dc_block_y 
                                   - (dc_block_y >>> 7);
                dc_block_x_prev <= $signed({1'b0, magnitude});
                dc_blocked_out  <= dc_block_y[12:0];
                dc_blocked_valid <= 1'b1;
            end
        end
    end


    // PWM DAC
    pwm_dac #(
        .INPUT_WIDTH(13),
        .PWM_BITS   (10)
    ) audio_dac (
        .clk        (clk),
        .reset      (reset_internal),
        .audio_in   (dc_blocked_out),
        .audio_valid(dc_blocked_valid),
        .pwm_out    (pwm_audio_out)
    );


    // LED Debug
    always_ff @(posedge clk) begin
        if (reset_internal)
            led <= '0;
        else if (dc_blocked_valid) begin
            // Display absolute value so both positive and negative swings light LEDs
            if (dc_blocked_out[12])
                led <= (~dc_blocked_out[11:4] + 1'b1);  // Negate if negative
            else
                led <= dc_blocked_out[11:4];
        end
    end

endmodule