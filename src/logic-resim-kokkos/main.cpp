// Kokkos port of logic-resim-cuda: event-driven logic circuit simulator.
// Original CUDA code: Simulation/src/sim/Simulator.cu
// Each parallel work-item simulates one gate across a full time window.

#include <Kokkos_Core.hpp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <random>

// Logic value encoding (matches original)
static constexpr char value0 = 0;
static constexpr char value1 = 1;
static constexpr char valueX = 2;
static constexpr char valueZ = 3;

// Truth-table store: each gate type occupies TRUTH_SIZE bytes.
// Bytes [0 .. TRUTH_SIZE-2] hold packed 2-bit output values;
// byte [TRUTH_SIZE-1] holds the number of input ports.
static constexpr int TRUTH_SIZE = 1025;

// Gate pin limits (matches original DelayMgr.h)
static constexpr int MAX_INPUT_PORT = 6;
static constexpr int MAX_DTUPLE     = 12;
static constexpr int MAX_DTABLE     = MAX_DTUPLE * MAX_INPUT_PORT;

typedef unsigned long long tUnit;
typedef unsigned int       dUnit;

// A single signal transition: time + logic value.
struct Event {
    tUnit t;
    char  v;
};

// Gate descriptor – uses flat indices into the shared event array instead of
// raw device pointers (makes it Kokkos-friendly).
struct SimGate {
    int   funcSer;                     // index of truth-table entry
    int   iPortBase[MAX_INPUT_PORT];   // base offset in all_input_events[]
    int   iPortLen [MAX_INPUT_PORT];   // # of events for this input wire
    dUnit dTable   [MAX_DTABLE];       // rise/fall delay table
};

// -------------------------------------------------------------------------
// Kokkos functor: one work-item per gate, mirrors simulateParallel() kernel.
// -------------------------------------------------------------------------
struct SimulateKernel {
    Kokkos::View<const SimGate*, Kokkos::MemoryTraits<Kokkos::RandomAccess>> gates;
    Kokkos::View<const Event*,   Kokkos::MemoryTraits<Kokkos::RandomAccess>> all_input_events;
    Kokkos::View<Event*>         output_events;
    Kokkos::View<const char*,    Kokkos::MemoryTraits<Kokkos::RandomAccess>> eTableMgr;
    Kokkos::View<const int*>     out_base;
    Kokkos::View<int*>           out_size;
    Kokkos::View<const int*>     out_max;
    Kokkos::View<int*>           overflow;
    tUnit dumpOff;
    int   simLimit;

    KOKKOS_INLINE_FUNCTION
    void operator()(const int gid) const {
        if (gid >= simLimit) return;

        const SimGate& gate = gates[gid];

        // Truth table and input length for this gate
        const char* eTable = &eTableMgr[gate.funcSer * TRUTH_SIZE];
        const int8_t inLen = static_cast<int8_t>(*(eTable + TRUTH_SIZE - 1));

        const int maxSize = out_max[gid];
        const int oBase   = out_base[gid];
        Event* oPort      = &output_events[oBase];

        overflow[gid] = 0;

        // Per-port current-event index and next-event time
        int   iPortIdx[MAX_INPUT_PORT];
        tUnit iTnext  [MAX_INPUT_PORT];

        for (int i = 0; i < inLen; i++) {
            iPortIdx[i] = 0;
            iTnext[i]   = (gate.iPortLen[i] > 1)
                              ? all_input_events[gate.iPortBase[i] + 1].t
                              : static_cast<tUnit>(-1);
        }

        // Build initial combined input pattern
        int16_t iVprev = 0;
        tUnit   currTime = static_cast<tUnit>(-1);
        bool    isInit   = true;

        for (int i = 0; i < inLen; i++) {
            iVprev  = static_cast<int16_t>(iVprev << 2);
            iVprev |= all_input_events[gate.iPortBase[i]].v;
            if (all_input_events[gate.iPortBase[i]].t > 0)
                isInit = false;
            if (iTnext[i] < currTime)
                currTime = iTnext[i];
        }

        int    currSize;
        int8_t oVprev;

        if (isInit) {
            currSize  = 1;
            oPort[0]  = {0, valueX};
            oVprev    = valueX;
        } else {
            currSize = out_size[gid];
            oVprev   = oPort[currSize - 1].v;
        }

        // Gate already fully simulated or no transitions within window
        if (currTime == static_cast<tUnit>(-1) || currTime > dumpOff) {
            if (oPort[currSize - 1].t != static_cast<tUnit>(-1)) {
                oPort[currSize] = {static_cast<tUnit>(-1), valueZ};
                out_size[gid]   = currSize + 1;
            }
            return;
        }

        // Simulation loop: advance through input transitions
        int16_t iVcurr = 0;
        int8_t  oVcurr = 0;
        dUnit   delay  = static_cast<dUnit>(-1);

        while (currTime != static_cast<tUnit>(-1) && currTime <= dumpOff) {
            // Update input pattern: advance each port that fires at currTime
            for (int i = 0; i < inLen; ++i) {
                if (iTnext[i] == currTime) {
                    ++iPortIdx[i];
                    const int next = iPortIdx[i] + 1;
                    iTnext[i] = (next < gate.iPortLen[i])
                                    ? all_input_events[gate.iPortBase[i] + next].t
                                    : static_cast<tUnit>(-1);
                }
                iVcurr  = static_cast<int16_t>(iVcurr << 2);
                iVcurr |= all_input_events[gate.iPortBase[i] + iPortIdx[i]].v;
            }

            // Evaluate truth table
            oVcurr = (eTable[iVcurr >> 2] >> ((iVcurr & 3) * 2)) & 3;

            if (oVcurr != oVprev) {
                // Determine delay-table index for this output transition
                int delayIdx;
                switch (oVprev << 2 | oVcurr) {
                    case 1: delayIdx = 0; break; // 0→1
                    case 4: delayIdx = 1; break; // 1→0
                    case 2: delayIdx = 2; break; // 0→X
                    case 9: delayIdx = 3; break; // X→1
                    case 6: delayIdx = 4; break; // 1→X
                    case 8: delayIdx = 5; break; // X→0
                    default: delayIdx = 0; break;
                }

                delay = static_cast<dUnit>(-1);
                for (int i = 0; i < inLen; ++i) {
                    int iTo   = (iVcurr >> ((inLen - i - 1) << 1)) & 3;
                    int iFrom = (iVprev >> ((inLen - i - 1) << 1)) & 3;
                    if (iTo != iFrom) {
                        dUnit d = (iFrom == 1 || iTo == 0)
                                      ? gate.dTable[MAX_DTUPLE * i + delayIdx + 6]
                                      : gate.dTable[MAX_DTUPLE * i + delayIdx];
                        if (d < delay)
                            delay = d;
                    }
                }

                currTime += delay;

                // Pop any later pending outputs (inertial delay model)
                while (currSize && oPort[currSize - 1].t >= currTime)
                    --currSize;

                // Record new output event
                if (!currSize || oPort[currSize - 1].v != oVcurr) {
                    oPort[currSize] = {currTime, static_cast<char>(oVcurr)};
                    ++currSize;
                    if (currSize == maxSize) {
                        overflow[gid]  = 1;
                        out_size[gid]  = currSize;
                        return;
                    }
                }
            }

            // Reset per-cycle accumulators
            iVprev  = iVcurr;
            oVprev  = oVcurr;
            iVcurr  = 0;
            oVcurr  = 0;
            currTime = static_cast<tUnit>(-1);
            delay    = static_cast<dUnit>(-1);

            for (int i = 0; i < inLen; ++i)
                if (iTnext[i] < currTime)
                    currTime = iTnext[i];
        }

        oPort[currSize] = {static_cast<tUnit>(-1), valueZ};
        out_size[gid]   = currSize + 1;
    }
};

// -------------------------------------------------------------------------
// Build a 2-input AND-gate truth table (with X/Z propagation).
// Encoding: iVcurr = (in0 << 2) | in1  (MSB = in0, LSB = in1)
// -------------------------------------------------------------------------
static void buildAndGateTruthTable(std::vector<char>& table) {
    table.assign(TRUTH_SIZE, 0);
    // 4^2 = 16 combinations, 2 bits each
    for (int iVcurr = 0; iVcurr < 16; ++iVcurr) {
        int in0 = (iVcurr >> 2) & 3;
        int in1 =  iVcurr       & 3;
        char out;
        if (in0 == value0 || in1 == value0) out = value0;
        else if (in0 == value1 && in1 == value1) out = value1;
        else out = valueX;
        int byte_pos = iVcurr >> 2;
        int bit_off  = (iVcurr & 3) * 2;
        table[byte_pos] |= static_cast<char>(out << bit_off);
    }
    table[TRUTH_SIZE - 1] = 2; // inLen = 2
}

// -------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        int simSize = 100;
        if (argc > 1) simSize = atoi(argv[1]);

        const int   maxInputPorts    = 2;     // 2-input AND gates
        const int   maxEventsPerWire = 64;    // cap per input wire
        const int   maxOutputEvents  = 256;   // cap per gate output
        const tUnit dumpOff          = 500000ULL;

        // Build truth table for one gate type (AND)
        std::vector<char> h_eTableMgr(TRUTH_SIZE);
        buildAndGateTruthTable(h_eTableMgr);

        const int totalInputEvents = simSize * maxInputPorts * maxEventsPerWire;
        const int totalOutputEvts  = simSize * maxOutputEvents;

        std::mt19937 rng(42);
        std::uniform_int_distribution<int>  val_dist(0, 1);
        std::uniform_int_distribution<int>  del_dist(100, 5000);

        std::vector<SimGate> h_gates(simSize);
        std::vector<Event>   h_input_events(totalInputEvents);
        std::vector<int>     h_out_base(simSize);
        std::vector<int>     h_out_max(simSize, maxOutputEvents);
        std::vector<int>     h_out_size(simSize, 1);

        int event_idx = 0;
        for (int g = 0; g < simSize; ++g) {
            h_gates[g].funcSer = 0;

            for (int p = 0; p < maxInputPorts; ++p) {
                h_gates[g].iPortBase[p] = event_idx;
                h_gates[g].iPortLen[p]  = maxEventsPerWire;

                tUnit t = 0;
                h_input_events[event_idx] = {0, static_cast<char>(val_dist(rng))};
                for (int e = 1; e < maxEventsPerWire - 1; ++e) {
                    t += static_cast<tUnit>(del_dist(rng));
                    h_input_events[event_idx + e] = {t, static_cast<char>(val_dist(rng))};
                }
                h_input_events[event_idx + maxEventsPerWire - 1] = {
                    static_cast<tUnit>(-1), valueZ};
                event_idx += maxEventsPerWire;
            }
            for (int p = maxInputPorts; p < MAX_INPUT_PORT; ++p) {
                h_gates[g].iPortBase[p] = 0;
                h_gates[g].iPortLen[p]  = 0;
            }

            for (int d = 0; d < MAX_DTABLE; ++d)
                h_gates[g].dTable[d] = static_cast<dUnit>(del_dist(rng));

            h_out_base[g] = g * maxOutputEvents;
        }

        // Allocate device views
        Kokkos::View<SimGate*> d_gates        ("gates",         simSize);
        Kokkos::View<Event*>   d_input_events ("input_events",  totalInputEvents);
        Kokkos::View<Event*>   d_output_events("output_events", totalOutputEvts);
        Kokkos::View<char*>    d_eTableMgr    ("eTableMgr",     TRUTH_SIZE);
        Kokkos::View<int*>     d_out_base     ("out_base",      simSize);
        Kokkos::View<int*>     d_out_max      ("out_max",       simSize);
        Kokkos::View<int*>     d_out_size     ("out_size",      simSize);
        Kokkos::View<int*>     d_overflow     ("overflow",      simSize);

        // Copy data to device via host mirrors
        {
            auto hm_gates = Kokkos::create_mirror_view(d_gates);
            auto hm_iev   = Kokkos::create_mirror_view(d_input_events);
            auto hm_etbl  = Kokkos::create_mirror_view(d_eTableMgr);
            auto hm_obase = Kokkos::create_mirror_view(d_out_base);
            auto hm_omax  = Kokkos::create_mirror_view(d_out_max);
            auto hm_osiz  = Kokkos::create_mirror_view(d_out_size);

            for (int g = 0; g < simSize; ++g) {
                hm_gates(g) = h_gates[g];
                hm_obase(g) = h_out_base[g];
                hm_omax(g)  = h_out_max[g];
                hm_osiz(g)  = h_out_size[g];
            }
            for (int e = 0; e < totalInputEvents; ++e) hm_iev(e) = h_input_events[e];
            for (int i = 0; i < TRUTH_SIZE; ++i)       hm_etbl(i) = h_eTableMgr[i];

            Kokkos::deep_copy(d_gates,         hm_gates);
            Kokkos::deep_copy(d_input_events,  hm_iev);
            Kokkos::deep_copy(d_eTableMgr,     hm_etbl);
            Kokkos::deep_copy(d_out_base,      hm_obase);
            Kokkos::deep_copy(d_out_max,       hm_omax);
            Kokkos::deep_copy(d_out_size,      hm_osiz);
        }
        Kokkos::deep_copy(d_overflow, 0);

        // Zero output event buffer
        Kokkos::deep_copy(d_output_events, Event{0, value0});

        printf("Logic re-simulation: %d 2-input AND gates, dumpOff=%llu\n",
               simSize, (unsigned long long)dumpOff);

        auto t0 = std::chrono::steady_clock::now();

        SimulateKernel kernel;
        kernel.gates            = d_gates;
        kernel.all_input_events = d_input_events;
        kernel.output_events    = d_output_events;
        kernel.eTableMgr        = d_eTableMgr;
        kernel.out_base         = d_out_base;
        kernel.out_size         = d_out_size;
        kernel.out_max          = d_out_max;
        kernel.overflow         = d_overflow;
        kernel.dumpOff          = dumpOff;
        kernel.simLimit         = simSize;

        Kokkos::parallel_for("simulateParallel",
                             Kokkos::RangePolicy<>(0, simSize), kernel);
        Kokkos::fence();

        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        printf("Total simulation time: %.6f (s)\n", elapsed);

        // Check overflow
        auto hm_ov   = Kokkos::create_mirror_view_and_copy(
                           Kokkos::HostSpace{}, d_overflow);
        auto hm_osiz = Kokkos::create_mirror_view_and_copy(
                           Kokkos::HostSpace{}, d_out_size);

        int overflowed = 0;
        long long total_events = 0;
        for (int g = 0; g < simSize; ++g) {
            overflowed  += hm_ov(g);
            total_events += hm_osiz(g);
        }
        printf("Overflowed: %d/%d gates, total output events: %lld\n",
               overflowed, simSize, total_events);
    }
    Kokkos::finalize();
    return 0;
}
