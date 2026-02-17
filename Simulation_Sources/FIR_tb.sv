`timescale 1ns/1ps


module FIR_tb;

    logic CLK = 0;
    logic RESET = 1;
    logic sample_valid = 0;

    logic signed [15:0] data_in;
    logic signed [31:0] data_out;
    logic data_valid;

    logic signed [15:0] input_mem [0:3999];
    integer outfile;

    // Clock generation
    always forever #5 CLK = ~CLK;  // 100 MHz

    // Instantiate DUT
    FIR dut (
        .data_in(data_in),
        .CLK(CLK),
        .RESET(RESET),
        .sample_valid(sample_valid),
        .data_out(data_out),
        .data_valid(data_valid)
    );

    initial begin

        $readmemh("input_signal.mem", input_mem);
        outfile = $fopen("output_signal.mem","w");
        if (outfile == 0)
            $fatal("File did not open.");

        #100;
        RESET = 0;

        for (int n = 0; n < 4000; n++) begin

         // Wait until FIR is idle
            wait(dut.busy == 0);

            data_in = input_mem[n];

            sample_valid = 1;
            @(posedge CLK);
            #10
            sample_valid = 0;

            @(posedge data_valid);

            $fwrite(outfile,"%08X\n", data_out);

        end

        $fclose(outfile);
        $stop;
    end

endmodule


