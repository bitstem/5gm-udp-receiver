#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>

#include <boost/asio.hpp>
#include <boost/exception/diagnostic_information.hpp>

const std::map<unsigned, unsigned> SAMPLE_RATE_FOR_PRB = {
    {  6,  1'920'000},
    { 15,  3'840'000},
    { 25,  7'680'000},
    { 30,  9'600'000},
    { 35, 11'520'000},
    { 40, 13'440'000},
    { 50, 15'360'000},
    { 75, 23'040'000},
    {100, 30'720'000}
};

static unsigned prb_for_sample_rate(unsigned rate) {
  for (const auto& kv : SAMPLE_RATE_FOR_PRB) {
    if (kv.second == rate) return kv.first;
  }
  return 0;
}

#define VERBOSE
#define BUFFERSIZE 2048

// undef the following for unicast reception
// #define MCAST_ADDRESS "239.1.1.22"

using boost::asio::ip::udp;

static inline int16_t read_be_i16(const unsigned char* p) {
  uint16_t u = (static_cast<uint16_t>(p[0]) << 8) | static_cast<uint16_t>(p[1]);
  return static_cast<int16_t>(u);
}

// Emulate: drop 4 LSB with rounding (16 -> 12), then later multiply by 2 (+6 dB).
static inline int16_t round_shift_right_4(int16_t x) {
  int32_t v = static_cast<int32_t>(x);
  // Round-to-nearest for signed two's complement.
  if (v >= 0) v += 8;
  else        v -= 8;
  v >>= 4;
  return static_cast<int16_t>(v);
}

struct WelfordMinMax {
  void reset() {
    n = 0;
    mean = 0.0;
    M2 = 0.0;
    minv = +std::numeric_limits<double>::infinity();
    maxv = -std::numeric_limits<double>::infinity();
  }
  void push(double x) {
    ++n;
    const double delta = x - mean;
    mean += delta / static_cast<double>(n);
    const double delta2 = x - mean;
    M2 += delta * delta2;
    if (x < minv) minv = x;
    if (x > maxv) maxv = x;
  }
  double stddev() const {
    if (n < 2) return 0.0;
    return std::sqrt(M2 / static_cast<double>(n - 1));
  }
  double p2p() const {
    if (n == 0) return 0.0;
    return maxv - minv;
  }
  uint64_t n = 0;
  double mean = 0.0;
  double M2 = 0.0;
  double minv = +std::numeric_limits<double>::infinity();
  double maxv = -std::numeric_limits<double>::infinity();
};

static inline double dbfs_power_from_meanp(double mean_p, double fullscale) {
  if (mean_p <= 0.0) return -std::numeric_limits<double>::infinity();
  return 10.0 * std::log10(mean_p / (fullscale * fullscale));
}

static inline double dbfs_rms_from_meanp(double mean_p, double fullscale) {
  // mean_p = E[I^2 + Q^2] in "counts^2"
  if (mean_p <= 0.0) return -std::numeric_limits<double>::infinity();
  const double rms = std::sqrt(mean_p);
  if (rms <= 0.0) return -std::numeric_limits<double>::infinity();
  return 20.0 * std::log10(rms / fullscale);
}

int main() {
  boost::asio::io_service io_service;
  boost::asio::ip::udp::socket socket(io_service);

  int32_t port = 31200;
  socket.open(boost::asio::ip::udp::v4());
  socket.set_option(udp::socket::reuse_address(true));
  socket.set_option(boost::asio::socket_base::receive_buffer_size(64 * 1024 * 1024));
  socket.bind(udp::endpoint(boost::asio::ip::address_v4::any(), port));

  {
    boost::asio::socket_base::receive_buffer_size opt;
    socket.get_option(opt);
    std::cout << "SO_RCVBUF = " << opt.value() << " bytes"
              << " (kernel may cap to net.core.rmem_max)\n";
  }

#ifdef MCAST_ADDRESS
  std::cout << "Joining multicast group " << MCAST_ADDRESS << std::endl;
  socket.set_option(
      boost::asio::ip::multicast::join_group(
          boost::asio::ip::address::from_string(MCAST_ADDRESS)));
#endif

  char buffer[BUFFERSIZE];
  udp::endpoint senderEndpoint;

  unsigned last_sample_offset = 0;
  unsigned last_ms = 0;
  unsigned last_s = 0;
  unsigned expected_increase = 320;
  unsigned nr_packets = 0;

  constexpr unsigned kComplexPerPacket = 320;
  constexpr double kFullScale = 32767.0;

  // Sample rate (and thus packets/ms) is auto-detected from the first observed
  // ms boundary: when samples wraps from some max back to 0 with ms incrementing,
  // sample_rate = (max + kComplexPerPacket) * 1000.
  unsigned last_offset_per_ms = 0;
  unsigned packets_per_ms = 0;
  bool rate_detected = false;

  // Detection state: only commit a detection after observing a full ms (start-to-start).
  bool detect_ms_started = false;
  unsigned detect_s = 0;
  unsigned detect_ms = 0;
  unsigned detect_max_samples = 0;

  // Near-rail threshold
  constexpr int16_t kNearRail = static_cast<int16_t>(0.98 * 32767.0);

  // CCDF thresholds relative to (running) RMS power estimate: RMS+8/10/12 dB in POWER ratios
  constexpr double kThrDbA = 8.0;
  constexpr double kThrDbB = 10.0;
  constexpr double kThrDbC = 12.0;
  const double kThrA = std::pow(10.0, kThrDbA / 10.0);
  const double kThrB = std::pow(10.0, kThrDbB / 10.0);
  const double kThrC = std::pow(10.0, kThrDbC / 10.0);

  // --- Per-second accumulators (RAW and EMULATED) ---
  bool sec_init = false;
  unsigned sec_s = 0;

  uint64_t sec_sum_p = 0;          // sum of p[n] over second (RAW)
  uint64_t sec_nsamp = 0;          // number of complex samples over second (RAW)
  uint64_t sec_rail_hits = 0;
  uint64_t sec_near_rail_hits = 0;

  uint64_t sec_sum_p_emul = 0;     // sum of p_emul[n] over second (EMUL)
  uint64_t sec_nsamp_emul = 0;     // number of complex samples over second (EMUL)

  // CCDF counters: count of samples above (running mean power * ratio)
  uint64_t sec_ccdf_a = 0, sec_ccdf_b = 0, sec_ccdf_c = 0;
  uint64_t sec_ccdf_a_emul = 0, sec_ccdf_b_emul = 0, sec_ccdf_c_emul = 0;

  uint64_t sec_max_p = 0;
  uint64_t sec_max_p_emul = 0;

  uint64_t sec_packets_received = 0;

  // --- Per-ms aggregation within each second (RAW and EMUL) ---
  bool ms_bucket_init = false;
  unsigned ms_bucket_s = 0, ms_bucket_ms = 0;

  uint64_t ms_sum_p = 0;           // sum of p[n] over this ms (RAW)
  uint64_t ms_nsamp = 0;           // samples in this ms (RAW)
  uint64_t ms_sum_p_emul = 0;      // sum of p_emul[n] over this ms (EMUL)
  uint64_t ms_nsamp_emul = 0;

  WelfordMinMax ms_db_stats;       // per-ms dBFS(power) stats over the last STREAM-second
  WelfordMinMax ms_db_stats_emul;  // same for emulated path

  auto reset_second_accums = [&]() {
    sec_sum_p = 0;
    sec_nsamp = 0;
    sec_rail_hits = 0;
    sec_near_rail_hits = 0;

    sec_sum_p_emul = 0;
    sec_nsamp_emul = 0;

    sec_ccdf_a = sec_ccdf_b = sec_ccdf_c = 0;
    sec_ccdf_a_emul = sec_ccdf_b_emul = sec_ccdf_c_emul = 0;

    sec_max_p = 0;
    sec_max_p_emul = 0;

    sec_packets_received = 0;

    // Reset per-ms stats for the new second
    ms_db_stats.reset();
    ms_db_stats_emul.reset();
  };

  auto reset_ms_bucket = [&](unsigned new_s, unsigned new_ms) {
    ms_bucket_s = new_s;
    ms_bucket_ms = new_ms;
    ms_sum_p = 0;
    ms_nsamp = 0;
    ms_sum_p_emul = 0;
    ms_nsamp_emul = 0;
  };

  // Initialize Welford stats
  ms_db_stats.reset();
  ms_db_stats_emul.reset();

  while (true) {
    try {
      const auto bytesRead =
          socket.receive_from(boost::asio::buffer(buffer, BUFFERSIZE), senderEndpoint);

      if (bytesRead == 0) continue;

      unsigned char* buf = reinterpret_cast<unsigned char*>(buffer);
      unsigned s = 0;
      unsigned ms = 0;
      unsigned samples = 0;

      bool null_flag = false;
      if (*buf & 0x4) null_flag = true;

      ms = (*buf++ & 0x3) << 8;
      ms |= (*buf++);

      samples = *buf++ << 8;
      samples |= *buf++;

      s = *buf++ << 24;
      s |= *buf++ << 16;
      s |= *buf++ << 8;
      s |= *buf++;

      // Guard against short packets before reading IQ.
      const size_t header_bytes =
          static_cast<size_t>(buf - reinterpret_cast<unsigned char*>(buffer));
      const size_t bytes_left = static_cast<size_t>(bytesRead) - header_bytes;
      const size_t needed_iq_bytes = static_cast<size_t>(kComplexPerPacket) * 4;
      if (bytes_left < needed_iq_bytes) {
        std::cout << "Short packet: got " << bytesRead << " bytes, need at least "
                  << (header_bytes + needed_iq_bytes) << " bytes\n";
        continue;
      }

      // Initialize second tracking
      if (!sec_init) {
        sec_s = s;
        sec_init = true;
        reset_second_accums();
      }

      // Initialize ms bucket
      if (!ms_bucket_init) {
        reset_ms_bucket(s, ms);
        ms_bucket_init = true;
      }

      // If ms bucket changes, finalize previous ms bucket and feed per-second ms stats
      if (s != ms_bucket_s || ms != ms_bucket_ms) {
        if (ms_nsamp > 0) {
          const double ms_mean_p = static_cast<double>(ms_sum_p) / static_cast<double>(ms_nsamp);
          const double ms_db = dbfs_power_from_meanp(ms_mean_p, kFullScale);
          if (std::isfinite(ms_db)) ms_db_stats.push(ms_db);
        }
        if (ms_nsamp_emul > 0) {
          const double ms_mean_p_emul =
              static_cast<double>(ms_sum_p_emul) / static_cast<double>(ms_nsamp_emul);
          const double ms_db_emul = dbfs_power_from_meanp(ms_mean_p_emul, kFullScale);
          if (std::isfinite(ms_db_emul)) ms_db_stats_emul.push(ms_db_emul);
        }
        reset_ms_bucket(s, ms);
      }

      // If second changes, print stats for previous second and reset.
      if (s != sec_s) {
        if (rate_detected) {
          const uint64_t expected = static_cast<uint64_t>(packets_per_ms) * 1000ULL;
          const uint64_t got = sec_packets_received;
          const double loss_pct =
              expected ? 100.0 * (1.0 - static_cast<double>(got) /
                                            static_cast<double>(expected))
                       : 0.0;
          const bool loss_warn = expected && got + (expected / 100) < expected; // >1% loss
          std::cout << (loss_warn ? "WARNING " : "")
                    << "SECOND s=" << sec_s
                    << " | packets " << got << "/" << expected
                    << " (loss " << std::fixed << std::setprecision(2) << loss_pct << "%)\n";
        }

        if (sec_nsamp > 0) {
          const double mean_p = static_cast<double>(sec_sum_p) / static_cast<double>(sec_nsamp);
          const double rms_dbfs = dbfs_rms_from_meanp(mean_p, kFullScale);
          const double p_dbfs = dbfs_power_from_meanp(mean_p, kFullScale);

          const double ccdf_a = static_cast<double>(sec_ccdf_a) / static_cast<double>(sec_nsamp);
          const double ccdf_b = static_cast<double>(sec_ccdf_b) / static_cast<double>(sec_nsamp);
          const double ccdf_c = static_cast<double>(sec_ccdf_c) / static_cast<double>(sec_nsamp);

          const double peak_db_over_rms =
              (mean_p > 0.0 && sec_max_p > 0)
                  ? 10.0 * std::log10(static_cast<double>(sec_max_p) / mean_p)
                  : 0.0;

          std::cout << "SECOND s=" << sec_s
                    << " | RAW: RMS " << std::fixed << std::setprecision(2) << rms_dbfs << " dBFS"
                    << " (P " << std::setprecision(2) << p_dbfs << " dBFS)"
                    << " peak_over_rms " << std::setprecision(2) << peak_db_over_rms << " dB"
                    << " CCDF(+" << kThrDbA << "dB)=" << std::setprecision(6) << ccdf_a
                    << " CCDF(+" << kThrDbB << "dB)=" << std::setprecision(6) << ccdf_b
                    << " CCDF(+" << kThrDbC << "dB)=" << std::setprecision(6) << ccdf_c
                    << " | per-ms std " << std::setprecision(3) << ms_db_stats.stddev() << " dB"
                    << " p2p " << std::setprecision(3) << ms_db_stats.p2p() << " dB"
                    << " | rail " << sec_rail_hits
                    << " near-rail " << sec_near_rail_hits
                    << "\n";
        }

        if (sec_nsamp_emul > 0) {
          const double mean_p_emul =
              static_cast<double>(sec_sum_p_emul) / static_cast<double>(sec_nsamp_emul);
          const double rms_dbfs_emul = dbfs_rms_from_meanp(mean_p_emul, kFullScale);
          const double p_dbfs_emul = dbfs_power_from_meanp(mean_p_emul, kFullScale);

          const double ccdf_a_emul =
              static_cast<double>(sec_ccdf_a_emul) / static_cast<double>(sec_nsamp_emul);
          const double ccdf_b_emul =
              static_cast<double>(sec_ccdf_b_emul) / static_cast<double>(sec_nsamp_emul);
          const double ccdf_c_emul =
              static_cast<double>(sec_ccdf_c_emul) / static_cast<double>(sec_nsamp_emul);

          const double peak_db_over_rms_emul =
              (mean_p_emul > 0.0 && sec_max_p_emul > 0)
                  ? 10.0 * std::log10(static_cast<double>(sec_max_p_emul) / mean_p_emul)
                  : 0.0;

          std::cout << "SECOND s=" << sec_s
                    << " | EMUL(12b+6dB): RMS " << std::fixed << std::setprecision(2)
                    << rms_dbfs_emul << " dBFS"
                    << " (P " << std::setprecision(2) << p_dbfs_emul << " dBFS)"
                    << " peak_over_rms " << std::setprecision(2) << peak_db_over_rms_emul << " dB"
                    << " CCDF(+" << kThrDbA << "dB)=" << std::setprecision(6) << ccdf_a_emul
                    << " CCDF(+" << kThrDbB << "dB)=" << std::setprecision(6) << ccdf_b_emul
                    << " CCDF(+" << kThrDbC << "dB)=" << std::setprecision(6) << ccdf_c_emul
                    << " | per-ms std " << std::setprecision(3) << ms_db_stats_emul.stddev() << " dB"
                    << " p2p " << std::setprecision(3) << ms_db_stats_emul.p2p() << " dB"
                    << "\n";
        }

        // Start new second
        sec_s = s;
        reset_second_accums();

        // Also reset ms stats for the new second; bucket continues normally
        // (ms_db_stats/ms_db_stats_emul already reset in reset_second_accums)
      }

      // --- Parse IQ and compute per-packet and per-sample stats ---
      uint64_t pkt_sum_p = 0;
      uint64_t pkt_sum_p_emul = 0;

      for (unsigned n = 0; n < kComplexPerPacket; ++n) {
        const int16_t i_s16 = read_be_i16(buf + 0);
        const int16_t q_s16 = read_be_i16(buf + 2);
        buf += 4;

        // Rail / near-rail (RAW)
        if (i_s16 == INT16_MIN || i_s16 == INT16_MAX) { ++sec_rail_hits; }
        if (q_s16 == INT16_MIN || q_s16 == INT16_MAX) { ++sec_rail_hits; }
        if (i_s16 >= kNearRail || i_s16 <= -kNearRail) { ++sec_near_rail_hits; }
        if (q_s16 >= kNearRail || q_s16 <= -kNearRail) { ++sec_near_rail_hits; }

        const int32_t i = static_cast<int32_t>(i_s16);
        const int32_t q = static_cast<int32_t>(q_s16);

        const uint64_t p = static_cast<uint64_t>(
            static_cast<int64_t>(i) * i + static_cast<int64_t>(q) * q);

        pkt_sum_p += p;
        if (p > sec_max_p) sec_max_p = p;

        // Emulate customer: 16->12 by dropping 4 LSB with rounding, then *2 (+6 dB)
        const int16_t i12 = round_shift_right_4(i_s16);
        const int16_t q12 = round_shift_right_4(q_s16);
        const int32_t i13 = static_cast<int32_t>(i12) << 1;
        const int32_t q13 = static_cast<int32_t>(q12) << 1;

        const uint64_t p_emul = static_cast<uint64_t>(
            static_cast<int64_t>(i13) * i13 + static_cast<int64_t>(q13) * q13);

        pkt_sum_p_emul += p_emul;
        if (p_emul > sec_max_p_emul) sec_max_p_emul = p_emul;

        // --- CCDF counting vs RUNNING mean power estimate for the second ---
        // Update running totals first (so threshold follows current estimate).
        // This makes CCDF approximate but stable enough for regression detection.
        sec_sum_p += p;
        ++sec_nsamp;
        sec_sum_p_emul += p_emul;
        ++sec_nsamp_emul;

        const double mean_p_run =
            static_cast<double>(sec_sum_p) / static_cast<double>(sec_nsamp);
        const double mean_p_emul_run =
            static_cast<double>(sec_sum_p_emul) / static_cast<double>(sec_nsamp_emul);

        if (mean_p_run > 0.0) {
          const double thrA = mean_p_run * kThrA;
          const double thrB = mean_p_run * kThrB;
          const double thrC = mean_p_run * kThrC;
          if (static_cast<double>(p) > thrA) ++sec_ccdf_a;
          if (static_cast<double>(p) > thrB) ++sec_ccdf_b;
          if (static_cast<double>(p) > thrC) ++sec_ccdf_c;
        }
        if (mean_p_emul_run > 0.0) {
          const double thrA = mean_p_emul_run * kThrA;
          const double thrB = mean_p_emul_run * kThrB;
          const double thrC = mean_p_emul_run * kThrC;
          if (static_cast<double>(p_emul) > thrA) ++sec_ccdf_a_emul;
          if (static_cast<double>(p_emul) > thrB) ++sec_ccdf_b_emul;
          if (static_cast<double>(p_emul) > thrC) ++sec_ccdf_c_emul;
        }
      }

      // Accumulate packet into current ms bucket (RAW + EMUL)
      ms_sum_p += pkt_sum_p;
      ms_nsamp += kComplexPerPacket;
      ms_sum_p_emul += pkt_sum_p_emul;
      ms_nsamp_emul += kComplexPerPacket;

#ifdef VERBOSE
      if (null_flag) {
        std::cout << "NULL packet s " << s << ", ms " << ms
                  << ", sample offset " << samples << " from "
                  << senderEndpoint.address().to_string() << "\n";
      }

      if (ms % 1000 == 0 && samples == 0) {
        uint64_t now =
            (std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::system_clock::now().time_since_epoch()))
                .count();
        uint64_t const DIFF_UTC_1970_1980 = 315'964'800U;
        now = (now - (DIFF_UTC_1970_1980 * 1000)) + 18000;

        int64_t delta =
            (int64_t)(((int64_t)s * 1000L) + ms) - (int64_t)now;

        std::cout << now << ": received " << bytesRead << " bytes for s " << s
                  << ", ms " << ms << ", sample offset " << samples
                  << " from sender " << senderEndpoint.address().to_string()
                  << ", " << delta << " ms delta, packet rate: "
                  << nr_packets / 1000.0 << "/ms\n";

        nr_packets = 0;
      }
#endif

      // Auto-detect sample rate (PRB count) by observing one full ms with no gaps:
      // start at samples=0, track max offset, commit on the next samples=0 if and only if
      // the ms is exactly +1 (consecutive) and the implied rate matches a known PRB count.
      if (!rate_detected) {
        const bool consecutive_ms =
            (s == detect_s && ms == detect_ms + 1) ||
            (s == detect_s + 1 && ms == 0 && detect_ms == 999);

        if (detect_ms_started && s == detect_s && ms == detect_ms) {
          if (samples > detect_max_samples) detect_max_samples = samples;
        } else if (detect_ms_started && samples == 0 && detect_max_samples > 0 &&
                   consecutive_ms) {
          const unsigned candidate_last = detect_max_samples;
          const unsigned sr = (candidate_last + kComplexPerPacket) * 1000;
          const unsigned prb = prb_for_sample_rate(sr);
          if (prb != 0) {
            last_offset_per_ms = candidate_last;
            packets_per_ms = (last_offset_per_ms / kComplexPerPacket) + 1;
            rate_detected = true;
            std::cout << "Detected sample rate " << sr << " Hz (PRB=" << prb
                      << ", " << packets_per_ms << " packets/ms, last offset/ms="
                      << last_offset_per_ms << ")\n";
          } else {
            // Implausible rate: probably saw a partial ms across a gap. Restart.
            detect_ms_started = (samples == 0);
            detect_s = s;
            detect_ms = ms;
            detect_max_samples = 0;
          }
        } else if (samples == 0) {
          // (Re)start detection at the start of an ms.
          detect_ms_started = true;
          detect_s = s;
          detect_ms = ms;
          detect_max_samples = 0;
        } else {
          // Mid-ms first observation, or non-consecutive ms — wait for next samples=0.
          detect_ms_started = false;
        }
      }

      // Timestamp continuity check (skipped until rate is known)
      if (rate_detected) {
        unsigned expected_s = last_s;
        unsigned expected_ms = last_ms;
        unsigned expected_samples = last_sample_offset + expected_increase;

        if (last_sample_offset == last_offset_per_ms) {
          expected_samples = 0;
          expected_ms++;
          if (last_ms == 999) {
            expected_ms = 0;
            expected_s++;
          }
        }

        if (ms != expected_ms || s != expected_s || samples != expected_samples) {
          std::cout << "Timestamp discontinuity detected! Expected "
                    << expected_s << "." << std::setfill('0') << std::setw(3)
                    << expected_ms << "." << std::setfill('0') << std::setw(4)
                    << expected_samples << ", got " << s << "."
                    << std::setfill('0') << std::setw(3) << ms << "."
                    << std::setfill('0') << std::setw(4) << samples << "\n";

          // Best effort: reset ms/second tracking to avoid smearing stats across gaps.
          // Keep rate_detected sticky — discontinuities are usually packet loss, not a
          // rate change. Only invalidate if we see hard evidence the rate is different
          // (samples beyond the current last-offset-per-ms).
          sec_init = false;
          ms_bucket_init = false;
          reset_second_accums();
          if (samples > last_offset_per_ms) {
            std::cout << "Sample offset " << samples << " > last_offset_per_ms "
                      << last_offset_per_ms << ", re-detecting rate\n";
            rate_detected = false;
            detect_ms_started = false;
          }
        }
      }

      last_sample_offset = samples;
      last_ms = ms;
      last_s = s;
      nr_packets++;
      sec_packets_received++;

    } catch (const boost::exception& e) {
      std::cerr << "Error. Reason: " << boost::diagnostic_information(e) << '\n';
    }
  }
}
