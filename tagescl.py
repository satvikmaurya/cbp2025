import numpy as np
from typing import Dict, List, Tuple, Optional
import array
import random

# Core constants
HISTBUFFERLENGTH = 4096   # Size of the history buffer

# TAGE parameters
NHIST = 36                # Number of history lengths
MINHIST = 6               # Minimum history length
MAXHIST = 3000            # Maximum history length
LOGG = 11                 # Log size of TAGE tables
TBITS = 9                 # Tag bits
LOGB = 13                 # Log size of bimodal table
CWIDTH = 3                # Counter width for TAGE tables
UWIDTH = 1                # Usefulness counter width
PHISTWIDTH = 27           # Path history width
LOGSIZEUSEALT = 4         # Size of useAlt table
ALTWIDTH = 5              # Width of useAlt counters
SIZEUSEALT = 1 << LOGSIZEUSEALT
BORN = 13                 # Threshold between low/high history lengths
BORNINFASSOC = 9          # Lower bound for 2-way associativity
BORNSUPASSOC = 23         # Upper bound for 2-way associativity
NBANKLOW = 15             # Banks for low history lengths
NBANKHIGH = 30            # Banks for high history lengths

# SC parameters
PERCWIDTH = 6             # Statistical corrector counter width
LOGGNB = 11               # Global branch GEHL
LOGPNB = 10               # Path/pattern history
LOGLNB = 11               # First local history
LOGSNB = 10               # Second local history
LOGTNB = 11               # Third local history
LOGBIAS = 8               # Bias tables

# Local history parameters
LOGLOCAL = 8
NLOCAL = 1 << LOGLOCAL
LOGSECLOCAL = 4
NSECLOCAL = 1 << LOGSECLOCAL
NTLOCAL = 16

# LOOP parameters
LOGL = 5
WIDTHNBITERLOOP = 10
LOOPTAG = 10

# Update threshold constants
LOGSIZEUP = 6
LOGSIZEUPS = LOGSIZEUP // 2
WIDTHRES = 12
WIDTHRESP = 8

# GEHL parameters
GNB = 3
PNB = 3
LNB = 3
SNB = 3
TNB = 2

class FoldedHistory:
    """Class to implement folded history for more efficient storage"""
    def __init__(self, original_length=0, compressed_length=0):
        self.comp = 0
        self.OLENGTH = original_length
        self.CLENGTH = compressed_length
        self.OUTPOINT = original_length % compressed_length if compressed_length > 0 else 0

    def init(self, original_length, compressed_length):
        self.comp = 0
        self.OLENGTH = original_length
        self.CLENGTH = compressed_length
        self.OUTPOINT = original_length % compressed_length

    def update(self, h, PT):
        """Update folded history with new branch outcome"""
        self.comp = (self.comp << 1) ^ h[PT & (HISTBUFFERLENGTH - 1)]
        self.comp ^= h[(PT + self.OLENGTH) & (HISTBUFFERLENGTH - 1)] << self.OUTPOINT
        self.comp ^= (self.comp >> self.CLENGTH)
        self.comp &= ((1 << self.CLENGTH) - 1)

class BimodalEntry:
    """Bimodal table entry"""
    def __init__(self):
        self.pred = 0     # Prediction counter
        self.hyst = 1     # Hysteresis bit

class TaggedEntry:
    """Tagged table entry for TAGE"""
    def __init__(self):
        self.ctr = 0     # Counter for prediction
        self.tag = 0     # Tag bits
        self.u = 0       # Usefulness counter

class LoopEntry:
    """Loop predictor entry"""
    def __init__(self):
        self.NbIter = 0         # Number of iterations
        self.confid = 0         # Confidence
        self.CurrentIter = 0    # Current iteration
        self.TAG = 0            # Tag
        self.age = 0            # Age counter
        self.dir = False        # Direction prediction

class BranchHistory:
    """History state management"""
    def __init__(self):
        # Global history
        self.GHIST = 0
        self.ghist = np.zeros(HISTBUFFERLENGTH, dtype=np.uint8)
        self.phist = 0          # Path history
        self.ptghist = 0        # Pointer in global history
        
        # Folded histories
        self.ch_i = [FoldedHistory() for _ in range(NHIST+1)]
        self.ch_t = [[FoldedHistory() for _ in range(NHIST+1)] for _ in range(2)]
        
        # Local histories
        self.L_shist = np.zeros(NLOCAL, dtype=np.uint64)
        self.S_slhist = np.zeros(NSECLOCAL, dtype=np.uint64)
        self.T_slhist = np.zeros(NTLOCAL, dtype=np.uint64)
        
        # Loop predictor
        self.ltable = [LoopEntry() for _ in range(1 << LOGL)]
        self.WITHLOOP = -1      # Loop predictor enable/disable

class TAGE_SC_L_Predictor:
    """Main predictor class implementing TAGE-SC-L"""
    def __init__(self):
        # Initialize state
        self.active_hist = BranchHistory()
        self.pred_time_histories = {}
        
        # Initialize tables
        self.btable = [BimodalEntry() for _ in range(1 << LOGB)]
        self.gtable = [[TaggedEntry() for _ in range(1 << LOGG)] for _ in range(NHIST+1)]
        
        # History length tables
        self.m = [0] * (NHIST + 1)
        self.TB = [0] * (NHIST + 1)
        self.logg = [0] * (NHIST + 1)
        
        # Set up history lengths
        self._initialize_history_lengths()
        
        # SC tables
        self.Bias = np.zeros(1 << LOGBIAS, dtype=np.int8)
        self.BiasSK = np.zeros(1 << LOGBIAS, dtype=np.int8)
        self.BiasBank = np.zeros(1 << LOGBIAS, dtype=np.int8)
        
        # Statistical corrector tables
        self.GGEHL = [np.zeros(1 << LOGGNB, dtype=np.int8) for _ in range(GNB)]
        self.PGEHL = [np.zeros(1 << LOGPNB, dtype=np.int8) for _ in range(PNB)]
        self.LGEHL = [np.zeros(1 << LOGLNB, dtype=np.int8) for _ in range(LNB)]
        self.SGEHL = [np.zeros(1 << LOGSNB, dtype=np.int8) for _ in range(SNB)]
        self.TGEHL = [np.zeros(1 << LOGTNB, dtype=np.int8) for _ in range(TNB)]
        
        # History lengths for GEHL components
        self.Gm = [40, 24, 10]  # Global branch history lengths
        self.Pm = [25, 16, 9]   # Path history lengths
        self.Lm = [11, 6, 3]    # Local history lengths
        self.Sm = [16, 11, 6]   # Second local history lengths
        self.Tm = [9, 4]        # Third local history lengths
        
        # Weights for SC components
        self.WG = np.full(1 << LOGSIZEUPS, 7, dtype=np.int8)
        self.WL = np.full(1 << LOGSIZEUPS, 7, dtype=np.int8)
        self.WS = np.full(1 << LOGSIZEUPS, 7, dtype=np.int8)
        self.WT = np.full(1 << LOGSIZEUPS, 7, dtype=np.int8)
        self.WP = np.full(1 << LOGSIZEUPS, 7, dtype=np.int8)
        self.WB = np.full(1 << LOGSIZEUPS, 4, dtype=np.int8)
        
        # Update threshold variables
        self.updatethreshold = 35 << 3
        self.Pupdatethreshold = np.zeros(1 << LOGSIZEUP, dtype=np.int32)
        
        # useAlt for TAGE
        self.use_alt_on_na = np.zeros(SIZEUSEALT, dtype=np.int8)
        
        # Statistical corrector choice counters
        self.FirstH = 0
        self.SecondH = 0
        
        # State for prediction
        self.GI = [0] * (NHIST + 1)     # Table indices
        self.GTAG = [0] * (NHIST + 1)   # Tags
        self.BI = 0                      # Bimodal index
        self.TICK = 0                    # Tick counter for resetting u bits
        self.LSUM = 0                    # Sum for statistical corrector
        
        # TAGE prediction state
        self.tage_pred = False           # TAGE prediction
        self.alttaken = False            # Alternate prediction
        self.LongestMatchPred = False    # Prediction from longest match
        self.HitBank = 0                 # Longest matching bank
        self.AltBank = 0                 # Alternate bank
        
        # Loop predictor state
        self.predloop = False            # Loop prediction
        self.LHIT = -1                   # Hit way in loop predictor
        self.LI = 0                      # Loop index
        self.LIB = 0                     # Loop index bits
        self.LTAG = 0                    # Loop tag
        self.LVALID = False              # Loop prediction validity
        
        # Prediction confidence
        self.HighConf = False            # High confidence flag
        self.LowConf = False             # Low confidence flag
        self.MedConf = False             # Medium confidence flag
        self.AltConf = False             # Alternate prediction confidence
        
        # For skipping tables
        self.NOSKIP = [True] * (NHIST + 1)
        self._initialize_noskip()
        
        # Seed for random number generation
        self.Seed = 0
        
        # Initialize folded histories
        self._initialize_folded_histories()
        
        # Bias the various tables
        self._initialize_bias_tables()

    def _initialize_history_lengths(self):
        """Set up geometric history lengths"""
        self.m[1] = MINHIST
        self.m[NHIST // 2] = MAXHIST
        
        # Compute geometric progression for first half
        for i in range(2, NHIST // 2 + 1):
            self.m[i] = int(MINHIST * pow(MAXHIST / MINHIST, (i - 1) / ((NHIST / 2) - 1)) + 0.5)
        
        # Copy to second half
        for i in range(NHIST, 1, -1):
            self.m[i] = self.m[(i + 1) // 2]
        
        # Set tag bits and log sizes
        for i in range(1, NHIST + 1):
            self.TB[i] = TBITS + 4 * (i >= BORN)
            self.logg[i] = LOGG
    
    def _initialize_noskip(self):
        """Initialize NOSKIP array for table usage"""
        for i in range(1, NHIST + 1):
            self.NOSKIP[i] = ((i - 1) & 1) or ((i >= BORNINFASSOC) and (i < BORNSUPASSOC))
        
        # Manually disable some entries to save storage (as in original code)
        self.NOSKIP[4] = False
        self.NOSKIP[NHIST - 2] = False
        self.NOSKIP[8] = False
        self.NOSKIP[NHIST - 6] = False
    
    def _initialize_folded_histories(self):
        """Initialize folded histories"""
        for i in range(1, NHIST + 1):
            self.active_hist.ch_i[i].init(self.m[i], self.logg[i])
            self.active_hist.ch_t[0][i].init(self.m[i], self.TB[i])
            self.active_hist.ch_t[1][i].init(self.m[i], self.TB[i] - 1)
    
    def _initialize_bias_tables(self):
        """Initialize the various bias tables with patterns"""
        for j in range(1 << LOGBIAS):
            # BiasSK initialization
            if j & 3 == 0:
                self.BiasSK[j] = -8
            elif j & 3 == 1:
                self.BiasSK[j] = 7
            elif j & 3 == 2:
                self.BiasSK[j] = -32
            else:
                self.BiasSK[j] = 31
                
            # Bias initialization
            if j & 3 == 0:
                self.Bias[j] = -32
            elif j & 3 == 1:
                self.Bias[j] = 31
            elif j & 3 == 2:
                self.Bias[j] = -1
            else:
                self.Bias[j] = 0
                
            # BiasBank initialization
            if j & 3 == 0:
                self.BiasBank[j] = -32
            elif j & 3 == 1:
                self.BiasBank[j] = 31
            elif j & 3 == 2:
                self.BiasBank[j] = -1
            else:
                self.BiasBank[j] = 0
        
        # Initialize GEHL tables with alternating values
        for i in range(GNB):
            for j in range(1 << LOGGNB):
                if not (j & 1):
                    self.GGEHL[i][j] = -1
                    
        for i in range(LNB):
            for j in range(1 << LOGLNB):
                if not (j & 1):
                    self.LGEHL[i][j] = -1
                    
        for i in range(SNB):
            for j in range(1 << LOGSNB):
                if not (j & 1):
                    self.SGEHL[i][j] = -1
                    
        for i in range(TNB):
            for j in range(1 << LOGTNB):
                if not (j & 1):
                    self.TGEHL[i][j] = -1
                    
        for i in range(PNB):
            for j in range(1 << LOGPNB):
                if not (j & 1):
                    self.PGEHL[i][j] = -1

    # Index computation functions
    def bindex(self, PC):
        """Compute bimodal table index"""
        return ((PC ^ (PC >> LOGB)) & ((1 << LOGB) - 1))
    
    def F(self, A, size, bank):
        """Function to mix path history bits"""
        A = A & ((1 << size) - 1)
        A1 = (A & ((1 << self.logg[bank]) - 1))
        A2 = (A >> self.logg[bank])
        
        if bank < self.logg[bank]:
            A2 = ((A2 << bank) & ((1 << self.logg[bank]) - 1)) + (A2 >> (self.logg[bank] - bank))
            
        A = A1 ^ A2
        
        if bank < self.logg[bank]:
            A = ((A << bank) & ((1 << self.logg[bank]) - 1)) + (A >> (self.logg[bank] - bank))
            
        return A
    
    def gindex(self, PC, bank, hist, ch_i):
        """Compute index for tagged tables"""
        M = min(self.m[bank], PHISTWIDTH)
        index = PC ^ (PC >> (abs(self.logg[bank] - bank) + 1)) ^ ch_i[bank].comp ^ self.F(hist, M, bank)
        return (index & ((1 << self.logg[bank]) - 1))
    
    def gtag(self, PC, bank, tag_0_array, tag_1_array):
        """Compute tag for tagged tables"""
        tag = PC ^ tag_0_array[bank].comp ^ (tag_1_array[bank].comp << 1)
        return (tag & ((1 << self.TB[bank]) - 1))
    
    def get_local_index(self, PC):
        """Get index for first local history"""
        return ((PC ^ (PC >> 2)) & (NLOCAL - 1))
    
    def get_second_local_index(self, PC):
        """Get index for second local history"""
        return (((PC ^ (PC >> 5))) & (NSECLOCAL - 1))
    
    def get_third_local_index(self, PC):
        """Get index for third local history"""
        return (((PC ^ (PC >> LOGTNB))) & (NTLOCAL - 1))
    
    def get_bias_index(self, PC):
        """Get index for bias table"""
        return (((((PC ^ (PC >> 2)) << 1) ^ (self.LowConf & (self.LongestMatchPred != self.alttaken))) << 1) + self.pred_inter) & ((1 << LOGBIAS) - 1)
    
    def get_biassk_index(self, PC):
        """Get index for BiasSK table"""
        return (((((PC ^ (PC >> (LOGBIAS-2))) << 1) ^ (self.HighConf)) << 1) + self.pred_inter) & ((1 << LOGBIAS) - 1)
    
    def get_biasbank_index(self, PC):
        """Get index for BiasBank table"""
        return (self.pred_inter + (((self.HitBank + 1) // 4) << 4) + (self.HighConf << 1) + 
                (self.LowConf << 2) + ((self.AltBank != 0) << 3) + ((PC ^ (PC >> 2)) << 7)) & ((1 << LOGBIAS) - 1)
    
    def get_indupd(self, PC):
        """Get index for update threshold table"""
        return (PC ^ (PC >> 2)) & ((1 << LOGSIZEUP) - 1)
    
    def get_indupds(self, PC):
        """Get index for weight tables"""
        return (PC ^ (PC >> 2)) & ((1 << LOGSIZEUPS) - 1)
    
    def get_indusealt(self):
        """Get index for useAlt table"""
        return ((((self.HitBank - 1) // 8) << 1) + self.AltConf) % (SIZEUSEALT - 1)
    
    # Prediction functions
    def getbim(self):
        """Get bimodal prediction"""
        BIM = (self.btable[self.BI].pred << 1) + (self.btable[self.BI >> 2].hyst)
        self.HighConf = (BIM == 0) or (BIM == 3)
        self.LowConf = not self.HighConf
        self.AltConf = self.HighConf
        self.MedConf = False
        return (self.btable[self.BI].pred > 0)
    
    def getloop(self, PC):
        """Get loop predictor prediction"""
        ltable = self.active_hist.ltable
        self.LHIT = -1
        
        self.LI = self.lindex(PC)
        self.LIB = ((PC >> (LOGL - 2)) & ((1 << (LOGL - 2)) - 1))
        self.LTAG = (PC >> (LOGL - 2)) & ((1 << 2 * LOOPTAG) - 1)
        self.LTAG ^= (self.LTAG >> LOOPTAG)
        self.LTAG = (self.LTAG & ((1 << LOOPTAG) - 1))
        
        for i in range(4):  # 4-way associative
            index = (self.LI ^ ((self.LIB >> i) << 2)) + i
            
            if ltable[index].TAG == self.LTAG:
                self.LHIT = i
                self.LVALID = ((ltable[index].confid == 15) or (ltable[index].confid * ltable[index].NbIter > 128))
                
                if ltable[index].CurrentIter + 1 == ltable[index].NbIter:
                    return not ltable[index].dir
                    
                return ltable[index].dir
        
        self.LVALID = False
        return False
    
    def lindex(self, PC):
        """Compute loop predictor index"""
        return (((PC ^ (PC >> 2)) & ((1 << (LOGL - 2)) - 1)) << 2)
    
    def MYRANDOM(self):
        """Simple PRNG for allocation decisions"""
        self.Seed += 1
        self.Seed ^= self.active_hist.phist
        self.Seed = ((self.Seed >> 21) + (self.Seed << 11)) & 0xFFFFFFFFFFFFFFFF
        self.Seed ^= self.active_hist.ptghist
        self.Seed = ((self.Seed >> 10) + (self.Seed << 22)) & 0xFFFFFFFFFFFFFFFF
        return (self.Seed & 0xFFFFFFFF)
    
    def ctrupdate(self, ctr, taken, nbits):
        """Update saturating counter"""
        if taken:
            if ctr < ((1 << (nbits - 1)) - 1):
                ctr += 1
        else:
            if ctr > -(1 << (nbits - 1)):
                ctr -= 1
        return ctr
    
    # Main prediction functions
    def Tagepred(self, PC):
        """TAGE prediction component"""
        self.HitBank = 0
        self.AltBank = 0
        
        # Compute indices and tags
        for i in range(1, NHIST + 1, 2):
            self.GI[i] = self.gindex(PC, i, self.active_hist.phist, self.active_hist.ch_i)
            self.GTAG[i] = self.gtag(PC, i, self.active_hist.ch_t[0], self.active_hist.ch_t[1])
            self.GTAG[i+1] = self.GTAG[i]
            self.GI[i+1] = self.GI[i] ^ (self.GTAG[i] & ((1 << LOGG) - 1))
        
        # Distribute tables among banks
        T = (PC ^ (self.active_hist.phist & ((1 << self.m[BORN]) - 1))) % NBANKHIGH
        for i in range(BORN, NHIST + 1):
            if self.NOSKIP[i]:
                self.GI[i] += (T << LOGG)
                T = (T + 1) % NBANKHIGH
        
        T = (PC ^ (self.active_hist.phist & ((1 << self.m[1]) - 1))) % NBANKLOW
        for i in range(1, BORN):
            if self.NOSKIP[i]:
                self.GI[i] += (T << LOGG)
                T = (T + 1) % NBANKLOW
        
        # Get bimodal prediction
        self.BI = self.bindex(PC)
        self.alttaken = self.getbim()
        self.tage_pred = self.alttaken
        self.LongestMatchPred = self.alttaken
        
        # Look for longest match
        for i in range(NHIST, 0, -1):
            if self.NOSKIP[i] and self.gtable[i][self.GI[i]].tag == self.GTAG[i]:
                self.HitBank = i
                self.LongestMatchPred = (self.gtable[i][self.GI[i]].ctr >= 0)
                break
        
        # Look for alternate prediction
        for i in range(self.HitBank - 1, 0, -1):
            if self.NOSKIP[i] and self.gtable[i][self.GI[i]].tag == self.GTAG[i]:
                self.AltBank = i
                self.alttaken = (self.gtable[i][self.GI[i]].ctr >= 0)
                break
        
        # Decide between longest match and alternate
        if self.HitBank > 0:
            if self.AltBank > 0:
                self.alttaken = (self.gtable[self.AltBank][self.GI[self.AltBank]].ctr >= 0)
                self.AltConf = (abs(2 * self.gtable[self.AltBank][self.GI[self.AltBank]].ctr + 1) > 1)
            else:
                self.alttaken = self.getbim()
            
            # Use alternate prediction in some cases
            use_alt_on_na = (self.use_alt_on_na[self.get_indusealt()] >= 0)
            if not use_alt_on_na or abs(2 * self.gtable[self.HitBank][self.GI[self.HitBank]].ctr + 1) > 1:
                self.tage_pred = self.LongestMatchPred
            else:
                self.tage_pred = self.alttaken
            
            # Set confidence levels
            self.HighConf = (abs(2 * self.gtable[self.HitBank][self.GI[self.HitBank]].ctr + 1) >= (1 << CWIDTH) - 1)
            self.LowConf = (abs(2 * self.gtable[self.HitBank][self.GI[self.HitBank]].ctr + 1) == 1)
            self.MedConf = (abs(2 * self.gtable[self.HitBank][self.GI[self.HitBank]].ctr + 1) == 5)
    
    def Gpredict(self, PC, BHIST, length, tab, NBR, logs, W):
        """GEHL component prediction"""
        PERCSUM = 0
        indupds = self.get_indupds(PC)
        
        for i in range(NBR):
            bhist = BHIST & ((1 << length[i]) - 1)
            
            # Index computation using a mix of PC and history
            index = (PC ^ bhist ^ (bhist >> (8 - i)) ^ (bhist >> (16 - 2 * i)) ^ 
                   (bhist >> (24 - 3 * i)) ^ (bhist >> (32 - 3 * i)) ^ 
                   (bhist >> (40 - 4 * i))) & ((1 << (logs - (i >= (NBR - 2)))) - 1)
            
            ctr = tab[i][index]
            PERCSUM += (2 * ctr + 1)
        
        # Apply variable threshold
        PERCSUM = (1 + (W[indupds] >= 0)) * PERCSUM
        return PERCSUM
    
    def predict(self, PC):
        """Main prediction function"""
        # TAGE prediction
        self.Tagepred(PC)
        pred_taken = self.tage_pred
        
        # Loop predictor
        self.predloop = self.getloop(PC)
        if self.active_hist.WITHLOOP >= 0 and self.LVALID:
            pred_taken = self.predloop
        
        # Statistical corrector prediction
        self.pred_inter = pred_taken
        
        # Compute SC prediction
        self.LSUM = 0
        
        # Add bias components
        ctr = self.Bias[self.get_bias_index(PC)]
        self.LSUM += (2 * ctr + 1)
        
        ctr = self.BiasSK[self.get_biassk_index(PC)]
        self.LSUM += (2 * ctr + 1)
        
        ctr = self.BiasBank[self.get_biasbank_index(PC)]
        self.LSUM += (2 * ctr + 1)
        
        # Apply variable threshold
        indupds = self.get_indupds(PC)
        self.LSUM = (1 + (self.WB[indupds] >= 0)) * self.LSUM
        
        # Add GEHL components
        self.LSUM += self.Gpredict((PC << 1) + self.pred_inter, self.active_hist.GHIST, 
                                  self.Gm, self.GGEHL, GNB, LOGGNB, self.WG)
        
        self.LSUM += self.Gpredict(PC, self.active_hist.phist, 
                                  self.Pm, self.PGEHL, PNB, LOGPNB, self.WP)
        
        # Add local history components
        local_idx = self.get_local_index(PC)
        self.LSUM += self.Gpredict(PC, self.active_hist.L_shist[local_idx], 
                                  self.Lm, self.LGEHL, LNB, LOGLNB, self.WL)
        
        seclocal_idx = self.get_second_local_index(PC)
        self.LSUM += self.Gpredict(PC, self.active_hist.S_slhist[seclocal_idx], 
                                  self.Sm, self.SGEHL, SNB, LOGSNB, self.WS)
        
        thirdlocal_idx = self.get_third_local_index(PC)
        self.LSUM += self.Gpredict(PC, self.active_hist.T_slhist[thirdlocal_idx], 
                                  self.Tm, self.TGEHL, TNB, LOGTNB, self.WT)
        
        # Final SC prediction
        SCPRED = (self.LSUM >= 0)
        
        # Calculate threshold for confidence
        indupd = self.get_indupd(PC)
        THRES = (self.updatethreshold >> 3) + self.Pupdatethreshold[indupd]
        THRES += 12 * ((self.WB[indupds] >= 0) + (self.WP[indupds] >= 0) + 
                     (self.WS[indupds] >= 0) + (self.WT[indupds] >= 0) + 
                     (self.WL[indupds] >= 0) + (self.WG[indupds] >= 0))
        
        # Choose between TAGE and SC prediction
        if self.pred_inter != SCPRED:
            pred_taken = SCPRED
            
            if self.HighConf:
                if abs(self.LSUM) < THRES // 4:
                    pred_taken = self.pred_inter
                elif abs(self.LSUM) < THRES // 2:
                    pred_taken = self.SecondH < 0 and SCPRED or self.pred_inter
            
            if self.MedConf and abs(self.LSUM) < THRES // 4:
                pred_taken = self.FirstH < 0 and SCPRED or self.pred_inter
        
        return pred_taken
    
    def update(self, PC, resolveDir):
        """Update predictor tables"""
        # Update loop predictor
        if self.LVALID:
            if self.predloop != self.pred_inter:
                self.active_hist.WITHLOOP = self.ctrupdate(self.active_hist.WITHLOOP, 
                                                        (self.predloop == resolveDir), 7)
        self.loopupdate(PC, resolveDir, (self.pred_taken != resolveDir))
        
        # Update SC choice counters
        if self.pred_inter != (self.LSUM >= 0):
            if abs(self.LSUM) < self.THRES:
                if self.HighConf and abs(self.LSUM) < self.THRES // 2 and abs(self.LSUM) >= self.THRES // 4:
                    self.SecondH = self.ctrupdate(self.SecondH, (self.pred_inter == resolveDir), 7)
                
                if self.MedConf and abs(self.LSUM) < self.THRES // 4:
                    self.FirstH = self.ctrupdate(self.FirstH, (self.pred_inter == resolveDir), 7)
        
        # Update threshold variables
        SCPRED = (self.LSUM >= 0)
        indupd = self.get_indupd(PC)
        
        if (SCPRED != resolveDir) or (abs(self.LSUM) < self.THRES):
            if SCPRED != resolveDir:
                self.Pupdatethreshold[indupd] += 1
                self.updatethreshold += 1
            else:
                self.Pupdatethreshold[indupd] -= 1
                self.updatethreshold -= 1
            
            # Clamp to range
            max_val = (1 << (WIDTHRESP - 1)) - 1
            min_val = -(1 << (WIDTHRESP - 1))
            
            self.Pupdatethreshold[indupd] = max(min(self.Pupdatethreshold[indupd], max_val), min_val)
            
            max_val = (1 << (WIDTHRES - 1)) - 1
            min_val = -(1 << (WIDTHRES - 1))
            
            self.updatethreshold = max(min(self.updatethreshold, max_val), min_val)
            
            # Update component weights
            indupds = self.get_indupds(PC)
            
            # Update bias weights
            bias_sum = ((2 * self.Bias[self.get_bias_index(PC)] + 1) + 
                       (2 * self.BiasSK[self.get_biassk_index(PC)] + 1) + 
                       (2 * self.BiasBank[self.get_biasbank_index(PC)] + 1))
            
            XSUM = self.LSUM - ((self.WB[indupds] >= 0) * bias_sum)
            
            if ((XSUM + bias_sum) >= 0) != (XSUM >= 0):
                self.WB[indupds] = self.ctrupdate(self.WB[indupds], ((bias_sum >= 0) == resolveDir), 6)
            
            # Update individual bias counters
            self.Bias[self.get_bias_index(PC)] = self.ctrupdate(self.Bias[self.get_bias_index(PC)], resolveDir, PERCWIDTH)
            self.BiasSK[self.get_biassk_index(PC)] = self.ctrupdate(self.BiasSK[self.get_biassk_index(PC)], resolveDir, PERCWIDTH)
            self.BiasBank[self.get_biasbank_index(PC)] = self.ctrupdate(self.BiasBank[self.get_biasbank_index(PC)], resolveDir, PERCWIDTH)
            
            # Update GEHL tables
            self.Gupdate((PC << 1) + self.pred_inter, resolveDir, self.active_hist.GHIST, 
                        self.Gm, self.GGEHL, GNB, LOGGNB, self.WG)
            
            self.Gupdate(PC, resolveDir, self.active_hist.phist, 
                        self.Pm, self.PGEHL, PNB, LOGPNB, self.WP)
            
            # Update local history tables
            local_idx = self.get_local_index(PC)
            self.Gupdate(PC, resolveDir, self.active_hist.L_shist[local_idx], 
                        self.Lm, self.LGEHL, LNB, LOGLNB, self.WL)
            
            seclocal_idx = self.get_second_local_index(PC)
            self.Gupdate(PC, resolveDir, self.active_hist.S_slhist[seclocal_idx], 
                        self.Sm, self.SGEHL, SNB, LOGSNB, self.WS)
            
            thirdlocal_idx = self.get_third_local_index(PC)
            self.Gupdate(PC, resolveDir, self.active_hist.T_slhist[thirdlocal_idx], 
                        self.Tm, self.TGEHL, TNB, LOGTNB, self.WT)
        
        # TAGE update
        ALLOC = ((self.tage_pred != resolveDir) and (self.HitBank < NHIST))
        
        # Don't allocate too often if overall prediction was correct
        if self.pred_taken == resolveDir and random.randint(0, 31) != 0:
            ALLOC = False
        
        # Managing longest matching entry
        if self.HitBank > 0:
            # Check if this is a pseudo-newly allocated entry
            PseudoNewAlloc = (abs(2 * self.gtable[self.HitBank][self.GI[self.HitBank]].ctr + 1) <= 1)
            
            if PseudoNewAlloc:
                if self.LongestMatchPred == resolveDir:
                    ALLOC = False
                
                if self.LongestMatchPred != self.alttaken:
                    indusealt = self.get_indusealt()
                    self.use_alt_on_na[indusealt] = self.ctrupdate(self.use_alt_on_na[indusealt], 
                                                               (self.alttaken == resolveDir), ALTWIDTH)
        
        # Allocate new entries if needed
        if ALLOC:
            T = 1  # Allocate 1+NNN entries (NNN is typically 1)
            
            A = 1
            if random.randint(0, 127) < 32:
                A = 2
                
            Penalty = 0
            NA = 0
            
            # Formula to choose between banks
            DEP = ((((self.HitBank - 1 + 2 * A) & 0xffe)) ^ (random.randint(0, 1)))
            
            for I in range(DEP, NHIST, 2):
                i = I + 1
                Done = False
                
                if self.NOSKIP[i]:
                    if self.gtable[i][self.GI[i]].u == 0:
                        # Replace only if confidence is low
                        if abs(2 * self.gtable[i][self.GI[i]].ctr + 1) <= 3:
                            self.gtable[i][self.GI[i]].tag = self.GTAG[i]
                            self.gtable[i][self.GI[i]].ctr = 0 if resolveDir else -1
                            NA += 1
                            
                            if T <= 0:
                                break
                                
                            I += 2
                            Done = True
                            T -= 1
                        else:
                            # Just slightly move the prediction
                            if self.gtable[i][self.GI[i]].ctr > 0:
                                self.gtable[i][self.GI[i]].ctr -= 1
                            else:
                                self.gtable[i][self.GI[i]].ctr += 1
                    else:
                        Penalty += 1
                
                if not Done:
                    i = (I ^ 1) + 1
                    if self.NOSKIP[i]:
                        if self.gtable[i][self.GI[i]].u == 0:
                            # Replace only if confidence is low
                            if abs(2 * self.gtable[i][self.GI[i]].ctr + 1) <= 3:
                                self.gtable[i][self.GI[i]].tag = self.GTAG[i]
                                self.gtable[i][self.GI[i]].ctr = 0 if resolveDir else -1
                                NA += 1
                                
                                if T <= 0:
                                    break
                                    
                                I += 2
                                T -= 1
                            else:
                                # Just slightly move the prediction
                                if self.gtable[i][self.GI[i]].ctr > 0:
                                    self.gtable[i][self.GI[i]].ctr -= 1
                                else:
                                    self.gtable[i][self.GI[i]].ctr += 1
                        else:
                            Penalty += 1
            
            # Manage u bit reset
            self.TICK += (Penalty - 2 * NA)
            self.TICK = max(0, self.TICK)
            
            if self.TICK >= 1024:  # BORNTICK
                # Reset u bits
                for i in range(1, BORN+1, BORN-1):
                    for j in range(len(self.gtable[i])):
                        self.gtable[i][j].u >>= 1
                
                self.TICK = 0
        
        # Update predictions
        if self.HitBank > 0:
            # Special case for weak counters
            if abs(2 * self.gtable[self.HitBank][self.GI[self.HitBank]].ctr + 1) == 1:
                if self.LongestMatchPred != resolveDir:
                    if self.AltBank > 0:
                        self.gtable[self.AltBank][self.GI[self.AltBank]].ctr = self.ctrupdate(
                            self.gtable[self.AltBank][self.GI[self.AltBank]].ctr, resolveDir, CWIDTH)
                    elif self.AltBank == 0:
                        self.baseupdate(resolveDir)
            
            # Update counter
            self.gtable[self.HitBank][self.GI[self.HitBank]].ctr = self.ctrupdate(
                self.gtable[self.HitBank][self.GI[self.HitBank]].ctr, resolveDir, CWIDTH)
            
            # Reset usefulness if prediction flipped
            if abs(2 * self.gtable[self.HitBank][self.GI[self.HitBank]].ctr + 1) == 1:
                self.gtable[self.HitBank][self.GI[self.HitBank]].u = 0
            
            # Special case for usefulness reset
            if (self.alttaken == resolveDir and 
                self.AltBank > 0 and 
                abs(2 * self.gtable[self.AltBank][self.GI[self.AltBank]].ctr + 1) == 7 and
                self.gtable[self.HitBank][self.GI[self.HitBank]].u == 1 and
                self.LongestMatchPred == resolveDir):
                self.gtable[self.HitBank][self.GI[self.HitBank]].u = 0
        else:
            # Update bimodal
            self.baseupdate(resolveDir)
        
        # Update usefulness counter
        if self.LongestMatchPred != self.alttaken and self.LongestMatchPred == resolveDir:
            if self.gtable[self.HitBank][self.GI[self.HitBank]].u < (1 << UWIDTH) - 1:
                self.gtable[self.HitBank][self.GI[self.HitBank]].u += 1
    
    def baseupdate(self, Taken):
        """Update bimodal table"""
        inter = (self.btable[self.BI].pred << 1) + self.btable[self.BI >> 2].hyst
        
        if Taken:
            if inter < 3:
                inter += 1
        elif inter > 0:
            inter -= 1
            
        self.btable[self.BI].pred = inter >> 1
        self.btable[self.BI >> 2].hyst = inter & 1
    
    def Gupdate(self, PC, taken, BHIST, length, tab, NBR, logs, W):
        """Update GEHL tables"""
        indupds = self.get_indupds(PC)
        PERCSUM = 0
        
        for i in range(NBR):
            bhist = BHIST & ((1 << length[i]) - 1)
            
            # Index computation using a mix of PC and history
            index = (PC ^ bhist ^ (bhist >> (8 - i)) ^ (bhist >> (16 - 2 * i)) ^ 
                   (bhist >> (24 - 3 * i)) ^ (bhist >> (32 - 3 * i)) ^ 
                   (bhist >> (40 - 4 * i))) & ((1 << (logs - (i >= (NBR - 2)))) - 1)
            
            PERCSUM += (2 * tab[i][index] + 1)
            tab[i][index] = self.ctrupdate(tab[i][index], taken, PERCWIDTH)
        
        # Update weight if component was decisive
        XSUM = self.LSUM - ((W[indupds] >= 0) * PERCSUM)
        if (XSUM + PERCSUM >= 0) != (XSUM >= 0):
            W[indupds] = self.ctrupdate(W[indupds], ((PERCSUM >= 0) == taken), 6)
    
    def loopupdate(self, PC, Taken, ALLOC):
        """Update loop predictor"""
        ltable = self.active_hist.ltable
        
        if self.LHIT >= 0:
            index = (self.LI ^ ((self.LIB >> self.LHIT) << 2)) + self.LHIT
            
            # Already a hit
            if self.LVALID:
                if Taken != self.predloop:
                    # Free the entry
                    ltable[index].NbIter = 0
                    ltable[index].age = 0
                    ltable[index].confid = 0
                    ltable[index].CurrentIter = 0
                    return
                elif ((self.predloop != self.tage_pred) or (random.randint(0, 7) == 0)):
                    if ltable[index].age < 15:  # CONFLOOP
                        ltable[index].age += 1
            
            ltable[index].CurrentIter += 1
            ltable[index].CurrentIter &= ((1 << WIDTHNBITERLOOP) - 1)
            
            if ltable[index].CurrentIter > ltable[index].NbIter:
                ltable[index].confid = 0
                ltable[index].NbIter = 0
            
            if Taken != ltable[index].dir:
                if ltable[index].CurrentIter == ltable[index].NbIter:
                    if ltable[index].confid < 15:
                        ltable[index].confid += 1
                    
                    if ltable[index].NbIter < 3:
                        # Free the entry
                        ltable[index].dir = Taken
                        ltable[index].NbIter = 0
                        ltable[index].age = 0
                        ltable[index].confid = 0
                else:
                    if ltable[index].NbIter == 0:
                        # First complete nest
                        ltable[index].confid = 0
                        ltable[index].NbIter = ltable[index].CurrentIter
                    else:
                        # Not same iterations as last time
                        ltable[index].NbIter = 0
                        ltable[index].confid = 0
                
                ltable[index].CurrentIter = 0
        
        elif ALLOC:
            X = random.randint(0, 3)
            
            if random.randint(0, 3) == 0:
                for i in range(4):
                    loop_hit_way_loc = (X + i) & 3
                    index = (self.LI ^ ((self.LIB >> loop_hit_way_loc) << 2)) + loop_hit_way_loc
                    
                    if ltable[index].age == 0:
                        ltable[index].dir = not Taken
                        ltable[index].TAG = self.LTAG
                        ltable[index].NbIter = 0
                        ltable[index].age = 7
                        ltable[index].confid = 0
                        ltable[index].CurrentIter = 0
                        break
                    else:
                        ltable[index].age -= 1
                        break

    def HistoryUpdate(self, PC, Taken, NextPC):
        """Update history registers"""
        # Update loop predictor validity
        if self.LVALID:
            if self.pred_taken != self.predloop:
                self.active_hist.WITHLOOP = self.ctrupdate(self.active_hist.WITHLOOP, 
                                                        (self.predloop == Taken), 7)
        
        # Global history update
        self.active_hist.GHIST = ((self.active_hist.GHIST << 1) + (Taken & (NextPC < PC))) & ((1 << MAXHIST) - 1)
        
        # Local history updates
        local_idx = self.get_local_index(PC)
        self.active_hist.L_shist[local_idx] = ((self.active_hist.L_shist[local_idx] << 1) + Taken) & ((1 << self.Lm[0]) - 1)
        
        seclocal_idx = self.get_second_local_index(PC)
        self.active_hist.S_slhist[seclocal_idx] = (((self.active_hist.S_slhist[seclocal_idx] << 1) + Taken) ^ (PC & 15)) & ((1 << self.Sm[0]) - 1)
        
        thirdlocal_idx = self.get_third_local_index(PC)
        self.active_hist.T_slhist[thirdlocal_idx] = ((self.active_hist.T_slhist[thirdlocal_idx] << 1) + Taken) & ((1 << self.Tm[0]) - 1)
        
        # Path history update
        T = ((PC ^ (PC >> 2))) ^ Taken
        PATH = PC ^ (PC >> 2) ^ (PC >> 4)
        
        for t in range(2):  # Typically 2 for conditional branches
            DIR = (T & 1)
            T >>= 1
            PATHBIT = (PATH & 127)
            PATH >>= 1
            
            # Update path/direction history
            self.active_hist.ptghist -= 1
            self.active_hist.ghist[self.active_hist.ptghist & (HISTBUFFERLENGTH - 1)] = DIR
            self.active_hist.phist = ((self.active_hist.phist << 1) ^ PATHBIT) & ((1 << PHISTWIDTH) - 1)
            
            # Update folded histories
            for i in range(1, NHIST + 1):
                self.active_hist.ch_i[i].update(self.active_hist.ghist, self.active_hist.ptghist)
                self.active_hist.ch_t[0][i].update(self.active_hist.ghist, self.active_hist.ptghist)
                self.active_hist.ch_t[1][i].update(self.active_hist.ghist, self.active_hist.ptghist)
    
    def predictSize(self):
        """Calculate predictor size in bits"""
        # TAGE size
        size = NBANKHIGH * (1 << self.logg[BORN]) * (CWIDTH + UWIDTH + self.TB[BORN])
        size += NBANKLOW * (1 << self.logg[1]) * (CWIDTH + UWIDTH + self.TB[1])
        
        # useAlt and misc tables
        size += SIZEUSEALT * ALTWIDTH
        size += (1 << LOGB) + (1 << (LOGB - 2))
        size += self.m[NHIST]  # Global history size
        size += PHISTWIDTH     # Path history width
        size += 10             # TICK counter
        
        # LOOP predictor
        size += (1 << LOGL) * (2 * WIDTHNBITERLOOP + LOOPTAG + 4 + 4 + 1)
        
        # Threshold tables
        size += WIDTHRES
        size += WIDTHRESP * (1 << LOGSIZEUP)
        size += 3 * 6 * (1 << LOGSIZEUPS)  # Weights
        
        # Bias tables
        size += PERCWIDTH * 3 * (1 << LOGBIAS)
        
        # GEHL components
        size += (GNB - 2) * (1 << LOGGNB) * PERCWIDTH + (1 << (LOGGNB - 1)) * (2 * PERCWIDTH)
        size += self.Gm[0]  # Global histories
        
        size += (PNB - 2) * (1 << LOGPNB) * PERCWIDTH + (1 << (LOGPNB - 1)) * (2 * PERCWIDTH)
        
        # Local histories
        size += (LNB - 2) * (1 << LOGLNB) * PERCWIDTH + (1 << (LOGLNB - 1)) * (2 * PERCWIDTH)
        size += NLOCAL * self.Lm[0]
        size += 6 * (1 << LOGSIZEUPS)
        
        # Second local history
        size += (SNB - 2) * (1 << LOGSNB) * PERCWIDTH + (1 << (LOGSNB - 1)) * (2 * PERCWIDTH)
        size += NSECLOCAL * self.Sm[0]
        size += 6 * (1 << LOGSIZEUPS)
        
        # Third local history
        size += (TNB - 2) * (1 << LOGTNB) * PERCWIDTH + (1 << (LOGTNB - 1)) * (2 * PERCWIDTH)
        size += NTLOCAL * self.Tm[0]
        size += 6 * (1 << LOGSIZEUPS)
        
        # Choice counters
        size += 2 * 7  # FirstH and SecondH
        
        return size


# Example usage function
def test_predictor():
    predictor = TAGE_SC_L_Predictor()
    print(f"Predictor size: {predictor.predictSize() / 8192:.2f} KB")
    
    # Example branch sequence
    branches = [
        (0x4000, True),
        (0x4100, False),
        (0x4200, True),
        (0x4000, True),
        (0x4100, False),
        (0x4000, True),
        (0x4100, True),  # Changed behavior
    ]
    
    correct = 0
    for i, (pc, actual) in enumerate(branches):
        prediction = predictor.predict(pc)
        
        # Determine next PC (normally provided by simulator)
        next_pc = pc + 4 if not actual else (pc + 400 if pc % 2 == 0 else pc - 400)
        
        # Update history right away
        predictor.HistoryUpdate(pc, actual, next_pc)
        
        # Update tables
        predictor.update(pc, actual)
        
        if prediction == actual:
            correct += 1
            
        print(f"Branch {i+1}: PC={hex(pc)}, Predicted={prediction}, Actual={actual}, "
              f"{'✓' if prediction == actual else '✗'}")
    
    print(f"Accuracy: {correct/len(branches):.2f}")

if __name__ == "__main__":
    test_predictor()