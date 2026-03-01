## Basys-3 Constraints File - v10
## AM SDR Radio - Stage 4: Full chain with noise gate

## Clock signal - 100 MHz
set_property -dict { PACKAGE_PIN W5   IOSTANDARD LVCMOS33 } [get_ports clk]
create_clock -add -name sys_clk_pin -period 10.00 -waveform {0 5} [get_ports clk]

## Reset Button (Center button)
set_property -dict { PACKAGE_PIN U18   IOSTANDARD LVCMOS33 } [get_ports reset]

##============================================================================
## XADC Analog Inputs - no IOSTANDARD on analog pins
##============================================================================
set_property PACKAGE_PIN L3 [get_ports Vauxp14]
set_property PACKAGE_PIN M3 [get_ports Vauxn14]
set_property PACKAGE_PIN J3 [get_ports Vauxp6]
set_property PACKAGE_PIN K3 [get_ports Vauxn6]

##============================================================================
## Local Oscillator Outputs - Pmod JB
##============================================================================
set_property -dict { PACKAGE_PIN A14   IOSTANDARD LVCMOS33 } [get_ports {cycle[0]}]
set_property -dict { PACKAGE_PIN A16   IOSTANDARD LVCMOS33 } [get_ports {cycle[1]}]

##============================================================================
## PWM Audio Output - Pmod JC pin K17
## Connect to RC lowpass filter for audio recovery:
##   1k + 100nF = ~1.6kHz cutoff (voice band)
##   1k + 10nF  = ~16kHz cutoff  (full audio band)
##============================================================================
set_property -dict { PACKAGE_PIN K17  IOSTANDARD LVCMOS33 } [get_ports pwm_audio_out]

##============================================================================
## LEDs
## Gate open (signal present):  shows DC-blocked audio level
## Gate closed (silence):       all LEDs off
##============================================================================
set_property -dict { PACKAGE_PIN U16   IOSTANDARD LVCMOS33 } [get_ports {led[0]}]
set_property -dict { PACKAGE_PIN E19   IOSTANDARD LVCMOS33 } [get_ports {led[1]}]
set_property -dict { PACKAGE_PIN U19   IOSTANDARD LVCMOS33 } [get_ports {led[2]}]
set_property -dict { PACKAGE_PIN V19   IOSTANDARD LVCMOS33 } [get_ports {led[3]}]
set_property -dict { PACKAGE_PIN W18   IOSTANDARD LVCMOS33 } [get_ports {led[4]}]
set_property -dict { PACKAGE_PIN U15   IOSTANDARD LVCMOS33 } [get_ports {led[5]}]
set_property -dict { PACKAGE_PIN U14   IOSTANDARD LVCMOS33 } [get_ports {led[6]}]
set_property -dict { PACKAGE_PIN V14   IOSTANDARD LVCMOS33 } [get_ports {led[7]}]

##============================================================================
## Configuration Options
##============================================================================
set_property CONFIG_VOLTAGE 3.3 [current_design]
set_property CFGBVS VCCO [current_design]
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
set_property BITSTREAM.CONFIG.CONFIGRATE 33 [current_design]
set_property CONFIG_MODE SPIx4 [current_design]
