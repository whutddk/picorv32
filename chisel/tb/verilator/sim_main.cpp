


#include <verilated.h>
#include "VSimTop.h"
#include <memory>
#include <iostream>
#include <getopt.h>

#include <sstream>



#if VM_TRACE
#include "verilated_fst_c.h"
#endif


char* img;
VSimTop *top;
#if VM_TRACE
VerilatedFstC* tfp;
#endif

vluint64_t main_time = 0;
vluint64_t main_cycle = 0;

double sc_time_stamp () {
	return main_time;
}


uint8_t flag_waveEnable = 0;
uint8_t flag_limitEnable = 0;

int prase_arg(int argc, char **argv) {
	int opt;
	while( -1 != ( opt = getopt( argc, argv, "lwf:" ) ) ) {
		switch(opt) {

			case 'w':
				flag_waveEnable = 1;
				std::cout << "Waveform is Enable" << std::endl;
				break;
			case 'l':
				flag_limitEnable = 1;
				break;
			case 'f':
				img = strdup(optarg);
				std::cout << "load in image is " << img << std::endl;
				break;
			case '?':
				std::cout << "-w to enable waveform" << std::endl;
				std::cout << "-f FILENAME to testfile" << std::endl;
				return -1;
				break;
			default:
				std::cout << opt << std::endl;
				assert(0);
		}
	}
	return 0;
}

static void sim_exit(){
#if VM_TRACE
	if ( flag_waveEnable ) { tfp->close(); }
#endif
	top->final();
	delete top;
}






int main(int argc, char **argv, char **env) {


	if ( -1 == prase_arg(argc, argv) ) {
		std::cout << "Prase Error." << std::endl;
		return -1;
	}

	char * temp[2];
	char cmd[64] = "+";
	strcat(cmd, img);
	// strcat(cmd, ".hex");
	temp[0] = "Verilated";
	temp[1] = cmd;
	char **argv_temp = temp;
	Verilated::commandArgs(2, argv_temp);		


	top = new VSimTop();

#if VM_TRACE
	tfp = new VerilatedFstC;
	if (flag_waveEnable) {
		Verilated::traceEverOn(true);
		top->trace(tfp, 99); // Trace 99 levels of hierarchy
		tfp->open("./build/wave.fst");		
	}
#endif

	
	top->reset = 1;
	top->clock = 0;

	while(!Verilated::gotFinish()) {
		static uint8_t flag_chk = 0;

		Verilated::timeInc(1);

		if ( main_time == 100 ){top->reset = 0;} //release reset

		if ( main_time % 10 == 1 ) { top->clock = 1; }       //generate clock posedge
		else if ( main_time % 10 == 6 ) { top->clock = 0; main_cycle ++; }  //generate clock negedge


		if(  main_time % 10 == 2 ) {//上升沿 延迟模拟
			if( main_time % 100000 == 9992 ) { top->irq = 1 << 4; } else if( 1 << 4 == top->irq ){ top->irq = 0; }
			if( main_time % 10000 == 702   ) { top->irq  = 1 << 5; } else if( 1 << 5 == top->irq ){ top->irq = 0; }
		}
			













#if VM_TRACE
		if ( flag_waveEnable ) { tfp->dump(Verilated::time()); }
#endif

		if ( flag_limitEnable ) {
			if ( main_cycle > 15000000 ){
				std::cout << "Timeout!!!!!" << std::endl;	
				sim_exit();
				return -1;
			} 			
		}



		if( main_time > 100 && main_time % 10 == 6 && top->trap ){ //已经解复位，下降沿， trap信号
			printf("TRAP after %1d clock cycles\n", main_cycle);

			if (top->tests_passed){
				printf("ALL TESTS PASSED.\n");
				sim_exit();
				return 0;
			} else {
				printf("ERROR!\n");
				sim_exit();
				return -1;
			}
		}


		main_time ++;
		top->eval();
	}

	
	sim_exit();
	return -1;


}



