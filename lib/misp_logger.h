#ifndef _MISP_LOGGER_H_
#define _MISP_LOGGER_H_

#include <fstream>
#include <string>
#include <cstdint>

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
            logfile << "seq_no,piece,pc,next_pc,branch_type,actual,predicted,";
            logfile << "global_hist,cycle,local_hist,confidence\n";
        }
    }
    
    ~MispredictionLogger() {
        if (enabled) logfile.close();
    }
    
    void log_misprediction(
        uint64_t seq_no, uint8_t piece, uint64_t pc, uint64_t next_pc,
        int branch_type, bool actual_outcome, bool predicted_outcome,
        uint64_t global_hist, uint64_t cycle,
        uint64_t local_hist = 0, int confidence = 0) {
        
        if (!enabled) return;
        
        logfile << seq_no << "," << (int)piece << ",0x" << std::hex << pc << ",0x" << next_pc << std::dec << ",";
        logfile << branch_type << "," << actual_outcome << "," << predicted_outcome << ",";
        logfile << "0x" << std::hex << global_hist << std::dec << "," << cycle << ",";
        logfile << "0x" << std::hex << local_hist << std::dec << "," << confidence << "\n";
        
        // Ensure data is flushed to file
        logfile.flush();
    }
};

extern MispredictionLogger* g_misp_logger;

#endif // _MISP_LOGGER_H_