#ifndef _MISP_LOGGER_H_
#define _MISP_LOGGER_H_

#include <fstream>
#include <string>
#include <cstdint>
#include <sys/types.h>

class MispredictionLogger {
private:
    std::ofstream logfile;
    bool enabled;
    
public:
    MispredictionLogger(const std::string& filename = "branch_misps.csv") {
        logfile.open(filename);
        enabled = logfile.is_open();
        
        // Write CSV header
        if (enabled) {
            logfile << "seq_no,piece,pc,next_pc,address,inst_type,actual,predicted\n";
        }
    }
    
    ~MispredictionLogger() {
        if (enabled) logfile.close();
    }
    
    void log_misprediction(
        uint64_t seq_no, uint8_t piece, uint64_t pc, uint64_t next_pc, uint64_t address,
        int inst_type, bool actual_outcome, bool predicted_outcome) {
        
        if (!enabled) return;
        
        logfile << seq_no << "," << (int)piece << ",0x" << std::hex << pc << ",0x" << next_pc << ",0x" << address << std::dec << ",";
        logfile << inst_type << "," << actual_outcome << "," << predicted_outcome << "\n";
        
        // Ensure data is flushed to file
        logfile.flush();
    }
};

extern MispredictionLogger* g_misp_logger;

#endif // _MISP_LOGGER_H_