module FIR (
    input logic signed [15:0] data_in,
    input logic CLK, 
    input logic RESET,
    input logic sample_valid,
    output logic data_valid,
    output logic signed [31:0] data_out
);

    logic signed [15:0] coeff_rom [0:127];
    logic signed [15:0] samples[0:127];
    logic signed [39:0] acc;
    logic [7:0] tap_index;
    logic busy;

    initial begin
        $readmemh("fir_coeffs.mem", coeff_rom);
    end
    
    always_ff @(posedge CLK) begin
        if(RESET) begin
            for(int i = 0; i < 128; i++)
                samples[i] <= 0;
             tap_index  <= 0;
             acc        <= 0;
             busy       <= 0;
             data_valid <= 0;
        end
        else begin
            data_valid <= 0;
            if(sample_valid && !busy) begin
                samples[0] <= data_in;
                for (int i = 1; i < 128; i++)
                    samples[i] <= samples[i-1];
                acc       <= 0;
                tap_index <= 0;
                busy      <= 1;
            end
            else if(busy) begin
                acc <= acc + samples[tap_index] * coeff_rom[tap_index];
                if (tap_index == 127) begin
                    data_out   <= acc >>> 15;
                    data_valid <= 1;
                    busy       <= 0;
                end
                else 
                    tap_index <= tap_index + 1;  
            end
        end   
      end   
endmodule