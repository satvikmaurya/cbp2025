#ifndef _PREDICTOR_H_
#define _PREDICTOR_H_

#include <bitset>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdlib.h>
#include <sys/types.h>
#include <unordered_map>
#include <vector>
#include <algorithm>

#define USE_PERCEPTRON
#define PERCEPTRON_SIZE 1024
#define GHIST_LEN 63

struct SampleHist
{
      uint64_t ghist;
      bool tage_pred;
      //
      SampleHist()
      {
          ghist = 0;
      }
};

#ifdef USE_PERCEPTRON
inline int8_t sat_add(int a, int b, int16_t lo = -127, int16_t hi = 127) {
    int32_t tmp = static_cast<int32_t>(a) + b;
    return static_cast<int8_t>(std::clamp(tmp, static_cast<int32_t>(lo),
                                           static_cast<int32_t>(hi)));
}

inline int8_t sgn(bool b) {return b ? 1 :-1;}

struct Result {
    bool prediction;
    int sum;
};

static std::unordered_map<uint64_t, Result> perceptron_history;

class perceptron {
public:
    static const int LHIST_LEN = 40;
    static const int NUM_WEIGHTS = GHIST_LEN + LHIST_LEN;
    std::bitset<LHIST_LEN> _lhist; // Local history register
    int8_t _weights[NUM_WEIGHTS];
    int8_t _bias;

    perceptron() : _bias(0) {
        _lhist.reset();
        for (int i = 0; i < NUM_WEIGHTS; ++i) {
            _weights[i] = 0;
        }
    }

    Result predict(std::bitset<GHIST_LEN> ghr) {
        int sum = _bias;
        for (int i = 0; i < GHIST_LEN; i++) {
            sum += (ghr[i] > 0) ? _weights[i] : -_weights[i];
        }
        for (int i = 0; i < LHIST_LEN; i++) {
            sum += (_lhist[i] > 0) ? _weights[GHIST_LEN + i] : -_weights[GHIST_LEN + i];
        }
        return {(sum >= 0), sum};
    }

    void update(std::bitset<GHIST_LEN> ghr, bool taken, bool tagescl, bool prediction) {
        // if(tagescl == taken) {
        //     _lhist = (_lhist << 1);
        //     _lhist.set(0, taken);
        //     return;
        // }

        int sum = _bias;
        for (int i = 0; i < GHIST_LEN; i++) {
            sum += (ghr[i] > 0) ? _weights[i] : -_weights[i];
        }
        for (int i = 0; i < LHIST_LEN; i++) {
            sum += (_lhist[i] > 0) ? _weights[GHIST_LEN + i] : -_weights[GHIST_LEN + i];
        }

        if(tagescl == taken) {
            _lhist = (_lhist << 1);
            _lhist.set(0, taken);
            return;
        }

        int direction = (taken) ? 1 : -1;
        _bias = sat_add(_bias, direction);
        for (int i = 0; i < GHIST_LEN; i++) {
            _weights[i] = sat_add(_weights[i], direction * ((ghr[i] > 0) ? 1 : -1));
        }
        for (int i = 0; i < LHIST_LEN; i++) {
            _weights[GHIST_LEN + i] = sat_add(_weights[GHIST_LEN + i], direction * ((_lhist[i] > 0) ? 1 : -1));
        }
        _lhist = (_lhist << 1);
        _lhist.set(0, taken);
    }
};

class perceptronPredictor {
public:
    std::bitset<GHIST_LEN> _ghr; // Global history register
    perceptron _table[PERCEPTRON_SIZE];

    perceptronPredictor() {
        _ghr.reset();
    }

    uint64_t hash(uint64_t PC) {
        uint64_t h = (PC >> 2);
        h = h ^ (h >> 16);
        h = h ^ (_ghr.to_ullong() & 0xFFF);
        return PC % PERCEPTRON_SIZE;
    }

    Result predict(uint64_t PC) {
        uint64_t pc_idx = hash(PC);
        return _table[pc_idx].predict(_ghr);
    }

    void update(uint64_t PC, bool taken, bool tagescl, bool prediction) {
        uint64_t pc_idx = hash(PC);
        _table[pc_idx].update(_ghr, taken, tagescl, prediction);
        _ghr = (_ghr << 1);
        _ghr.set(0, taken);
    }
};

class metaPerceptron {
    public:
        static const int NUM_WEIGHTS = 2;
        int8_t _weights[NUM_WEIGHTS];
        int8_t _bias;
        int8_t sat_cnt;
    
        metaPerceptron() : _bias(0), sat_cnt(0) {
            for (int i = 0; i < NUM_WEIGHTS; ++i) {
                _weights[i] = 0;
            }
        }
    
        void update(int8_t* features, bool taken, bool prediction, bool perceptron_pred) {
            int tagescl = sgn(features[0]);
            int psign = sgn(perceptron_pred);

            if(prediction == features[0]) {
                sat_cnt = std::max(0, sat_cnt - 1);
            } else if(prediction == perceptron_pred) {
                sat_cnt = std::min(3, sat_cnt + 1);
            }

            int direction = (taken) ? 1 : -1;

            if(tagescl == psign && prediction != taken) return;

            _weights[0] = sat_add(_weights[0], direction * tagescl);
            _weights[1] = sat_add(_weights[1], direction * psign);
            _bias = sat_add(_bias, direction);
        }
         
        bool predict(int8_t* features, bool perceptron_pred) {
            return (sat_cnt < 2);
            int sum = _bias;

            int tagescl = sgn(features[0]);
            int psign = sgn(perceptron_pred);

            sum += (tagescl * _weights[0]) +
                   (psign * _weights[1]);
            return (sum >= 0);
        }
    };

class metaPredictor {
public:
    metaPerceptron _table[PERCEPTRON_SIZE];
    std::bitset<GHIST_LEN> _ghr; // consider this a copy of the GHR for the specPerceptron, not counted for storage
    uint64_t count; // just for debug

    metaPredictor() : count(0) {
        _ghr.reset();
    }

    uint64_t hash(uint64_t PC) {
        uint64_t h = (PC >> 2);
        h = h ^ (h >> 16);
        h = h ^ (_ghr.to_ullong() & 0xFFF);
        return PC % PERCEPTRON_SIZE;
    }

    bool predict(uint64_t PC, int8_t* features, bool perceptron_pred) {
        uint64_t pc_idx = hash(PC);
        bool prediction = _table[pc_idx].predict(features, perceptron_pred);
        return prediction;
    }

    void update(uint64_t PC, int8_t* features, bool taken, bool prediction, bool perceptron_pred) {
        count++;
        if(taken == prediction) {
            _ghr = (_ghr << 1);
            _ghr.set(0, taken);
            return;
        }

        uint64_t pc_idx = hash(PC);
        _table[pc_idx].update(features, taken, prediction, perceptron_pred);
        _ghr = (_ghr << 1);
        _ghr.set(0, taken);
    }
};
#endif

class SampleCondPredictor
{
        SampleHist active_hist;
        std::unordered_map<uint64_t/*key*/, SampleHist/*val*/> pred_time_histories;
#ifdef USE_PERCEPTRON
        perceptronPredictor perceptron_predictor;
        metaPredictor meta_predictor;
#endif
    public:

        SampleCondPredictor (void)
        {
        }

        void setup()
        {
        }

        void terminate()
        {
        }

        // sample function to get unique instruction id
        uint64_t get_unique_inst_id(uint64_t seq_no, uint8_t piece) const
        {
            assert(piece < 16);
            return (seq_no << 4) | (piece & 0x000F);
        }

        bool predict (uint64_t seq_no, uint8_t piece, uint64_t PC, const bool tage_pred, std::vector<int8_t>& features)
        {
            active_hist.tage_pred = tage_pred;
            // checkpoint current hist
            pred_time_histories.emplace(get_unique_inst_id(seq_no, piece), active_hist);
            const bool pred_taken = predict_using_given_hist(seq_no, piece, PC, active_hist, true/*pred_time_predict*/);
#ifdef USE_PERCEPTRON
            Result res = perceptron_predictor.predict(PC);
            perceptron_history[get_unique_inst_id(seq_no, piece)] = res;
            bool t = meta_predictor.predict(PC, features.data(), res.prediction);
            return t ? pred_taken : res.prediction;
#else
            return pred_taken;
#endif
        }

        bool predict_using_given_hist (uint64_t seq_no, uint8_t piece, uint64_t PC, const SampleHist& hist_to_use, const bool pred_time_predict)
        {
            return hist_to_use.tage_pred;
        }

        void history_update (uint64_t seq_no, uint8_t piece, uint64_t PC, bool taken, uint64_t nextPC)
        {
            active_hist.ghist = active_hist.ghist << 1;
            if(taken)
            {
                active_hist.ghist |= 1;
            }
        }

        void update (uint64_t seq_no, uint8_t piece, uint64_t PC, bool resolveDir, bool predDir, uint64_t nextPC, std::vector<int8_t>& features)
        {
            const auto pred_hist_key = get_unique_inst_id(seq_no, piece);
            const auto& pred_time_history = pred_time_histories.at(pred_hist_key);
            update(PC, resolveDir, predDir, nextPC, pred_time_history);
            pred_time_histories.erase(pred_hist_key);
#ifdef USE_PERCEPTRON
            auto it = perceptron_history.find(get_unique_inst_id(seq_no, piece));
            if(it != perceptron_history.end()) {
                perceptron_predictor.update(PC, resolveDir, features[0], it->second.prediction);
                meta_predictor.update(PC, features.data(), resolveDir, predDir, it->second.prediction);
                perceptron_history.erase(it);
            } else {
                exit(1);
            }
#endif
        }

        void update (uint64_t PC, bool resolveDir, bool pred_taken, uint64_t nextPC, const SampleHist& hist_to_use)
        {
        }
};
// =================
// Predictor End
// =================

#endif
static SampleCondPredictor cond_predictor_impl;
