
// Module: xadc_reader
// Description: XADC interface using DRP + AXI Stream output mode
//              Simplified data_valid logic - fires on every AXIS sample
//              rather than waiting for a matched I+Q pair.
//
//              i_data and q_data are always both output together, but one will
//              be one conversion cycle old (~5us at 200kSPS). This is negligible
//              for audio baseband signals and eliminates any pairing starvation.
//
// AXIS channel IDs (Xilinx XADC):
//   vauxp6/vauxn6   -> m_axis_tid = 0x16
//   vauxp14/vauxn14 -> m_axis_tid = 0x1E

module xadc_reader #(
    parameter int DATA_WIDTH = 12
)(
    input  logic        clk,
    input  logic        reset,

    // Analog differential inputs
    input  logic        Vauxp14,
    input  logic        Vauxn14,
    input  logic        Vauxp6,
    input  logic        Vauxn6,

    // Sampled outputs
    output logic [DATA_WIDTH-1:0] i_data,
    output logic [DATA_WIDTH-1:0] q_data,
    output logic                  data_valid,  // Pulses every time either channel updates

    // Diagnostic
    output logic        eoc,
    output logic        eos
);

    // AXI Stream signals
    logic [15:0] m_axis_tdata;
    logic        m_axis_tvalid;
    logic [4:0]  m_axis_tid;

    // Channel IDs
    localparam logic [4:0] CH_VAUX14 = 5'h1E;  // I channel
    localparam logic [4:0] CH_VAUX6  = 5'h16;  // Q channel


    // XADC Wizard Instantiation
    xadc_wiz_0 xadc_inst (
        .m_axis_aclk    (clk),
        .s_axis_aclk    (clk),
        .m_axis_resetn  (~reset),

        .m_axis_tvalid  (m_axis_tvalid),
        .m_axis_tready  (1'b1),
        .m_axis_tdata   (m_axis_tdata),
        .m_axis_tid     (m_axis_tid),

        // DRP - tied off
        .di_in          (16'h0),
        .daddr_in       (7'h0),
        .den_in         (1'b0),
        .dwe_in         (1'b0),
        .drdy_out       (),
        .do_out         (),

        .vp_in          (1'b0),
        .vn_in          (1'b0),
        .vauxp6         (Vauxp6),
        .vauxn6         (Vauxn6),
        .vauxp14        (Vauxp14),
        .vauxn14        (Vauxn14),

        .eoc_out        (eoc),
        .eos_out        (eos),
        .channel_out    (),
        .alarm_out      (),
        .busy_out       ()
    );

    // Sample Capture
    // Register whichever channel just arrived, hold the other at its last value
    // Fire data_valid on every incoming sample (I or Q)
    // Ignore any channel IDs that aren't our two aux channels (temp, vccint etc)
    
    always_ff @(posedge clk) begin
        if (reset) begin
            i_data     <= '0;
            q_data     <= '0;
            data_valid <= 1'b0;
        end else begin
            data_valid <= 1'b0;

            if (m_axis_tvalid) begin
                case (m_axis_tid)
                    CH_VAUX14: begin
                        i_data     <= m_axis_tdata[15:4];
                        data_valid <= 1'b1;
                    end
                    CH_VAUX6: begin
                        q_data     <= m_axis_tdata[15:4];
                        data_valid <= 1'b1;
                    end
                    default: ;  // Ignore temperature, vccint etc
                endcase
            end
        end
    end

endmodule
