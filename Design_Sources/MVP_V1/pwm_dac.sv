// Module: pwm_dac
// Description: 10-bit PWM DAC for audio output
//              PWM frequency = 100MHz / 1024 = ~97.6 kHz
//              New audio samples are only latched at the START of each PWM
//              period to prevent glitches on the output waveform.
//
//              Input is signed (centred around 0). We add a DC offset to
//              convert to unsigned before comparing against the PWM counter.
//
//              Resolution: 10 bits = 1024 levels
//              PWM period: 1024 clock cycles = 10.24 us
//              PWM frequency: ~97.6 kHz (well above audio band)
//
// Output behaviour:
//   Silence (audio_in = 0):  50% duty cycle - correct midpoint
//   Positive peak:           ~100% duty cycle
//   Negative peak:           ~0% duty cycle

module pwm_dac #(
    parameter int INPUT_WIDTH = 13,    // Signed input width
    parameter int PWM_BITS    = 10     // PWM resolution (1024 levels)
)(
    input  logic                          clk,
    input  logic                          reset,

    input  logic signed [INPUT_WIDTH-1:0] audio_in,
    input  logic                          audio_valid,

    output logic                          pwm_out
);

    localparam int PWM_LEVELS = 1 << PWM_BITS;           // 1024
    localparam int PWM_MID    = 1 << (PWM_BITS - 1);    // 512 = silence = 50% duty

    // PWM counter - free running 0 to PWM_LEVELS-1
    logic [PWM_BITS-1:0] pwm_counter;

    // Latched threshold - only updated at start of PWM period
    logic [PWM_BITS-1:0] pwm_threshold;

    // Pending sample - written any time audio_valid fires
    logic [PWM_BITS-1:0] pending_threshold;
    logic                pending_valid;

    //==========================================================================
    // Convert signed audio input to unsigned PWM threshold
    // Input range: -2^(INPUT_WIDTH-1) to +2^(INPUT_WIDTH-1)-1
    // We shift right to fit into PWM_BITS, then add midpoint offset
    //==========================================================================
    logic [PWM_BITS-1:0] threshold_next;

    always_comb begin
        // Arithmetic shift right to scale from INPUT_WIDTH to PWM_BITS
        // Then add PWM_MID to centre around 50% duty
        automatic logic signed [INPUT_WIDTH-1:0] scaled;
        scaled = audio_in >>> (INPUT_WIDTH - PWM_BITS);
        threshold_next = $unsigned(scaled[PWM_BITS-1:0]) + PWM_MID[PWM_BITS-1:0];
    end

    //==========================================================================
    // Capture incoming audio sample into pending register
    //==========================================================================
    always_ff @(posedge clk) begin
        if (reset) begin
            pending_threshold <= PWM_MID[PWM_BITS-1:0];
            pending_valid     <= 1'b0;
        end else if (audio_valid) begin
            pending_threshold <= threshold_next;
            pending_valid     <= 1'b1;
        end
    end

    //==========================================================================
    // PWM counter - free running
    //==========================================================================
    always_ff @(posedge clk) begin
        if (reset)
            pwm_counter <= '0;
        else
            pwm_counter <= pwm_counter + 1;
    end

    //==========================================================================
    // Latch pending threshold at start of PWM period (counter == 0)
    // This prevents mid-period threshold changes that cause output glitches
    //==========================================================================
    always_ff @(posedge clk) begin
        if (reset) begin
            pwm_threshold <= PWM_MID[PWM_BITS-1:0];
        end else if (pwm_counter == '0) begin
            if (pending_valid)
                pwm_threshold <= pending_threshold;
            // If no new sample, hold last value (natural for audio)
        end
    end

    //==========================================================================
    // PWM output comparison
    //==========================================================================
    always_ff @(posedge clk) begin
        if (reset)
            pwm_out <= 1'b0;
        else
            pwm_out <= (pwm_counter < pwm_threshold);
    end

endmodule
