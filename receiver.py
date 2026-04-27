#!/usr/bin/env python3
"""
UDP IQ test receiver with live time-domain power and SA-style spectrum display.

Packet format (matches main.cpp):
    byte0:    bit2 (0x4) = NULL flag; (byte0 & 0x3) = ms high 2 bits
    byte1:    ms low 8 bits
    byte2-3:  sample offset within the ms (big-endian uint16)
    byte4-7:  stream second s (big-endian uint32)
    payload:  320 complex samples, I/Q each int16 BE  (1280 bytes)

Features:
- Auto-detects sample rate from the PRB lookup table
  (1.92 / 3.84 / 7.68 / 9.6 / 11.52 / 13.44 / 15.36 / 23.04 / 30.72 MHz).
- Per-ms / per-second metrics: RMS dBFS, peak/RMS, CCDF at +8/+10/+12 dB,
  rail and near-rail counts, and the 12-bit-drop+x2 ('EMUL') path that
  mirrors main.cpp.
- Per-second packet-loss check vs. the detected packets/ms rate.
- Live plot:
    top    -- per-ms power timeline (RAW + EMUL)
    bottom -- SA-style PSD with min-hold / avg / max-hold traces
- Press 'c' on the plot window to clear the min/max hold.

Deps: numpy, scipy, matplotlib
"""

import argparse
import math
import socket
import struct
import sys
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import welch


SAMPLE_RATE_FOR_PRB = {
    6:    1_920_000,
    15:   3_840_000,
    25:   7_680_000,
    30:   9_600_000,
    35:  11_520_000,
    40:  13_440_000,
    50:  15_360_000,
    75:  23_040_000,
    100: 30_720_000,
}
PRB_FOR_SAMPLE_RATE = {v: k for k, v in SAMPLE_RATE_FOR_PRB.items()}

K_COMPLEX_PER_PACKET = 320
FULL_SCALE = 32767.0
NEAR_RAIL = int(0.98 * 32767)
CCDF_DB = (8.0, 10.0, 12.0)
CCDF_RATIOS = [10.0 ** (d / 10.0) for d in CCDF_DB]

BUFFERSIZE = 2048
SO_RCVBUF_BYTES = 64 * 1024 * 1024


# ---------------- CLI ----------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--port", type=int, default=31200)
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--mcast-group", default=None,
                   help="multicast group to join (e.g. 239.1.1.22)")
    p.add_argument("--rbw", type=float, default=100e3,
                   help="resolution bandwidth for spectrum (Hz)")
    p.add_argument("--fc", type=float, default=0.0,
                   help="center frequency for axis labelling (Hz)")
    p.add_argument("--hold", type=float, default=2.0,
                   help="min/max-hold window for spectrum (s)")
    p.add_argument("--history", type=float, default=10.0,
                   help="time-domain power-plot history (s)")
    p.add_argument("--spectrum-interval", type=float, default=0.2,
                   help="seconds between spectrum recomputes")
    p.add_argument("--no-emul", action="store_true",
                   help="skip the 12-bit-emulation accounting/trace")
    p.add_argument("--no-spectrum", action="store_true",
                   help="disable spectrum panel (less CPU)")
    p.add_argument("--quiet", action="store_true",
                   help="suppress per-second stdout summaries")
    return p.parse_args()


# ---------------- Math helpers ----------------

def dbfs_power_from_meanp(mean_p: float) -> float:
    if mean_p <= 0.0:
        return float("-inf")
    return 10.0 * math.log10(mean_p / (FULL_SCALE * FULL_SCALE))


def dbfs_rms_from_meanp(mean_p: float) -> float:
    if mean_p <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(math.sqrt(mean_p) / FULL_SCALE)


def round_shift_right_4(arr_i32: np.ndarray) -> np.ndarray:
    """Drop 4 LSB with signed round-to-nearest: (v +/- 8) >> 4."""
    v = np.where(arr_i32 >= 0, arr_i32 + 8, arr_i32 - 8)
    return v >> 4


class WelfordMinMax:
    __slots__ = ("n", "mean", "M2", "minv", "maxv")

    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.minv = float("inf")
        self.maxv = float("-inf")

    def push(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)
        if x < self.minv:
            self.minv = x
        if x > self.maxv:
            self.maxv = x

    def stddev(self) -> float:
        if self.n < 2:
            return 0.0
        return math.sqrt(self.M2 / (self.n - 1))

    def p2p(self) -> float:
        if self.n == 0:
            return 0.0
        return self.maxv - self.minv


# ---------------- Sample-rate auto-detect ----------------

class SampleRateDetector:
    """Mirror main.cpp: observe one full ms (samples=0..max..0 with consecutive
    ms) and look the implied rate up in the PRB table."""

    def __init__(self):
        self.detected = False
        self.sample_rate = None
        self.prb = None
        self.last_offset_per_ms = None
        self.packets_per_ms = None
        self._started = False
        self._s = 0
        self._ms = 0
        self._max_samples = 0

    def feed(self, s: int, ms: int, samples: int) -> bool:
        """Returns True if detection just completed on this packet."""
        if self.detected:
            return False

        consecutive = (
            (s == self._s and ms == self._ms + 1) or
            (s == self._s + 1 and ms == 0 and self._ms == 999)
        )

        if self._started and s == self._s and ms == self._ms:
            if samples > self._max_samples:
                self._max_samples = samples
            return False

        if (self._started and samples == 0 and self._max_samples > 0
                and consecutive):
            sr = (self._max_samples + K_COMPLEX_PER_PACKET) * 1000
            prb = PRB_FOR_SAMPLE_RATE.get(sr)
            if prb is not None:
                self.sample_rate = sr
                self.prb = prb
                self.last_offset_per_ms = self._max_samples
                self.packets_per_ms = self._max_samples // K_COMPLEX_PER_PACKET + 1
                self.detected = True
                print(f"Detected sample rate {sr} Hz "
                      f"(PRB={prb}, {self.packets_per_ms} packets/ms, "
                      f"last offset/ms={self.last_offset_per_ms})",
                      flush=True)
                return True
            # implausible rate: restart
            self._started = True
            self._s, self._ms, self._max_samples = s, ms, 0
            return False

        if samples == 0:
            self._started = True
            self._s, self._ms, self._max_samples = s, ms, 0
        else:
            self._started = False
        return False

    def invalidate(self):
        self.detected = False
        self.sample_rate = None
        self.prb = None
        self.last_offset_per_ms = None
        self.packets_per_ms = None
        self._started = False


# ---------------- Spectrum (Welch + min/avg/max hold) ----------------

class Spectrum:
    """Welch PSD on a rolling IQ buffer; keeps a deque of recent traces for
    min-hold / avg / max-hold display."""

    def __init__(self, rbw_hz: float, hold_seconds: float,
                 update_seconds: float):
        self.rbw_hz = rbw_hz
        self.hold_seconds = hold_seconds
        self.update_seconds = update_seconds
        self.fs = None
        self._iq = None
        self._iq_size = 0
        self._iq_head = 0
        self._iq_filled = 0
        self._chunk_samples = 0
        self._nperseg = 0
        n_traces = max(1, int(round(hold_seconds / update_seconds)))
        self._traces = deque(maxlen=n_traces)
        self.f = None
        self._last_update = 0.0

    def set_sample_rate(self, fs: float):
        self.fs = fs
        # Hann window ENBW ~= 1.5 bins; pick nperseg so RBW ~= ENBW * fs / nperseg.
        enbw = 1.5
        self._nperseg = max(64, int(round(enbw * fs / self.rbw_hz)))
        # Compute on a chunk that comfortably exceeds nperseg so Welch can
        # average a few segments per trace. Cap at ~5 ms to bound CPU.
        self._chunk_samples = min(int(0.005 * fs),
                                  max(self._nperseg * 4, int(0.002 * fs)))
        self._iq_size = max(self._chunk_samples * 2, self._nperseg * 4)
        self._iq = np.zeros(self._iq_size, dtype=np.complex64)
        self._iq_head = 0
        self._iq_filled = 0
        self._traces.clear()
        self.f = None
        self._last_update = 0.0

    def push(self, iq: np.ndarray):
        if self._iq is None:
            return
        n = iq.shape[0]
        if n >= self._iq_size:
            self._iq[:] = iq[-self._iq_size:]
            self._iq_head = 0
            self._iq_filled = self._iq_size
            return
        end = self._iq_head + n
        if end <= self._iq_size:
            self._iq[self._iq_head:end] = iq
        else:
            first = self._iq_size - self._iq_head
            self._iq[self._iq_head:] = iq[:first]
            self._iq[:n - first] = iq[first:]
        self._iq_head = end % self._iq_size
        self._iq_filled = min(self._iq_size, self._iq_filled + n)

    def maybe_update(self, now: float) -> bool:
        if self._iq is None or self._iq_filled < self._chunk_samples:
            return False
        if now - self._last_update < self.update_seconds:
            return False
        self._last_update = now

        end = self._iq_head
        start = (end - self._chunk_samples) % self._iq_size
        if start < end:
            chunk = self._iq[start:end]
        else:
            chunk = np.concatenate((self._iq[start:], self._iq[:end]))

        f, pxx = welch(
            chunk, fs=self.fs, window="hann",
            nperseg=self._nperseg,
            noverlap=self._nperseg // 2,
            return_onesided=False, detrend=False, scaling="density",
        )
        self.f = np.fft.fftshift(f)
        trace = 10.0 * np.log10(np.fft.fftshift(pxx) * self.rbw_hz + 1e-30)
        self._traces.append(trace)
        return True

    def get_min_avg_max(self):
        if not self._traces or self.f is None:
            return None
        T = np.stack(self._traces)
        return T.min(axis=0), T.mean(axis=0), T.max(axis=0)

    def clear_hold(self):
        self._traces.clear()

    def n_traces(self) -> int:
        return len(self._traces)


# ---------------- Receiver / metrics ----------------

class Metrics:
    def __init__(self, args):
        self.args = args
        self.rate = SampleRateDetector()
        self.spectrum = (None if args.no_spectrum else
                         Spectrum(args.rbw, args.hold, args.spectrum_interval))

        # Second tracking
        self.sec_init = False
        self.sec_s = 0
        self.sec_packets_received = 0

        # Per-second accumulators (RAW)
        self.sec_sum_p = 0.0
        self.sec_nsamp = 0
        self.sec_rail = 0
        self.sec_near = 0
        self.sec_max_p = 0.0
        self.sec_ccdf = [0, 0, 0]

        # Per-second accumulators (EMUL)
        self.sec_sum_p_e = 0.0
        self.sec_nsamp_e = 0
        self.sec_max_p_e = 0.0
        self.sec_ccdf_e = [0, 0, 0]

        # Per-ms bucket
        self.ms_bucket_init = False
        self.ms_bucket_s = 0
        self.ms_bucket_ms = 0
        self.ms_sum_p = 0.0
        self.ms_nsamp = 0
        self.ms_sum_p_e = 0.0
        self.ms_nsamp_e = 0

        self.ms_stats = WelfordMinMax()
        self.ms_stats_e = WelfordMinMax()

        # Plot buffers
        plot_points = max(1000, int(args.history * 1000))
        self.plot_t = deque(maxlen=plot_points)
        self.plot_raw = deque(maxlen=plot_points)
        self.plot_emul = deque(maxlen=plot_points)

        # Timestamp continuity
        self.last_s = None
        self.last_ms = None
        self.last_samples = None

    def reset_second(self, new_s: int):
        self.sec_s = new_s
        self.sec_packets_received = 0
        self.sec_sum_p = 0.0
        self.sec_nsamp = 0
        self.sec_rail = 0
        self.sec_near = 0
        self.sec_max_p = 0.0
        self.sec_ccdf = [0, 0, 0]
        self.sec_sum_p_e = 0.0
        self.sec_nsamp_e = 0
        self.sec_max_p_e = 0.0
        self.sec_ccdf_e = [0, 0, 0]
        self.ms_stats.reset()
        self.ms_stats_e.reset()

    def reset_ms_bucket(self, s: int, ms: int):
        self.ms_bucket_s = s
        self.ms_bucket_ms = ms
        self.ms_sum_p = 0.0
        self.ms_nsamp = 0
        self.ms_sum_p_e = 0.0
        self.ms_nsamp_e = 0

    def hard_reset(self):
        self.sec_init = False
        self.ms_bucket_init = False
        self.last_s = self.last_ms = self.last_samples = None
        self.ms_stats.reset()
        self.ms_stats_e.reset()

    def _finalize_ms_bucket(self):
        if self.ms_nsamp > 0:
            db = dbfs_power_from_meanp(self.ms_sum_p / self.ms_nsamp)
            if math.isfinite(db):
                self.ms_stats.push(db)
            self.plot_t.append(time.time())
            self.plot_raw.append(db)
            if not self.args.no_emul and self.ms_nsamp_e > 0:
                db_e = dbfs_power_from_meanp(self.ms_sum_p_e / self.ms_nsamp_e)
                if math.isfinite(db_e):
                    self.ms_stats_e.push(db_e)
                self.plot_emul.append(db_e)
            else:
                self.plot_emul.append(float("nan"))

    def _print_second_summary(self):
        if self.args.quiet:
            return

        if self.rate.detected:
            expected = self.rate.packets_per_ms * 1000
            got = self.sec_packets_received
            loss_pct = 100.0 * (1.0 - got / expected) if expected else 0.0
            warn = "WARNING " if got + (expected // 100) < expected else ""
            print(f"{warn}SECOND s={self.sec_s} | packets {got}/{expected} "
                  f"(loss {loss_pct:.2f}%)", flush=True)

        if self.sec_nsamp > 0:
            meanp = self.sec_sum_p / self.sec_nsamp
            rms_dbfs = dbfs_rms_from_meanp(meanp)
            p_dbfs = dbfs_power_from_meanp(meanp)
            peak_over_rms = (10.0 * math.log10(self.sec_max_p / meanp)
                             if (meanp > 0 and self.sec_max_p > 0) else 0.0)
            ccdf = [c / self.sec_nsamp for c in self.sec_ccdf]
            print(
                f"SECOND s={self.sec_s} | RAW: RMS {rms_dbfs:6.2f} dBFS "
                f"(P {p_dbfs:6.2f} dBFS) peak_over_rms {peak_over_rms:5.2f} dB "
                f"CCDF(+{CCDF_DB[0]:.0f})={ccdf[0]:.6f} "
                f"CCDF(+{CCDF_DB[1]:.0f})={ccdf[1]:.6f} "
                f"CCDF(+{CCDF_DB[2]:.0f})={ccdf[2]:.6f} "
                f"| per-ms std {self.ms_stats.stddev():.3f} dB "
                f"p2p {self.ms_stats.p2p():.3f} dB "
                f"| rail {self.sec_rail} near-rail {self.sec_near}",
                flush=True,
            )

        if not self.args.no_emul and self.sec_nsamp_e > 0:
            meanp = self.sec_sum_p_e / self.sec_nsamp_e
            rms_dbfs = dbfs_rms_from_meanp(meanp)
            p_dbfs = dbfs_power_from_meanp(meanp)
            peak_over_rms = (10.0 * math.log10(self.sec_max_p_e / meanp)
                             if (meanp > 0 and self.sec_max_p_e > 0) else 0.0)
            ccdf = [c / self.sec_nsamp_e for c in self.sec_ccdf_e]
            print(
                f"SECOND s={self.sec_s} | EMUL(12b+6dB): RMS {rms_dbfs:6.2f} dBFS "
                f"(P {p_dbfs:6.2f} dBFS) peak_over_rms {peak_over_rms:5.2f} dB "
                f"CCDF(+{CCDF_DB[0]:.0f})={ccdf[0]:.6f} "
                f"CCDF(+{CCDF_DB[1]:.0f})={ccdf[1]:.6f} "
                f"CCDF(+{CCDF_DB[2]:.0f})={ccdf[2]:.6f} "
                f"| per-ms std {self.ms_stats_e.stddev():.3f} dB "
                f"p2p {self.ms_stats_e.p2p():.3f} dB",
                flush=True,
            )

    def process_packet(self, s, ms, samples, null_flag, iq_bytes):
        # Sample-rate detection (sticky once detected)
        rate_just_detected = self.rate.feed(s, ms, samples)
        if rate_just_detected and self.spectrum is not None:
            self.spectrum.set_sample_rate(self.rate.sample_rate)

        # Timestamp continuity vs. detected rate
        if self.rate.detected and self.last_s is not None:
            expected_s = self.last_s
            expected_ms = self.last_ms
            expected_samples = self.last_samples + K_COMPLEX_PER_PACKET
            if self.last_samples == self.rate.last_offset_per_ms:
                expected_samples = 0
                expected_ms += 1
                if self.last_ms == 999:
                    expected_ms = 0
                    expected_s += 1

            if (s, ms, samples) != (expected_s, expected_ms, expected_samples):
                print(
                    f"Timestamp discontinuity! "
                    f"Expected {expected_s}.{expected_ms:03d}.{expected_samples:04d}, "
                    f"got {s}.{ms:03d}.{samples:04d}",
                    flush=True,
                )
                self.hard_reset()
                if samples > (self.rate.last_offset_per_ms or 0):
                    print("Sample offset beyond detected last_offset_per_ms; "
                          "re-detecting rate", flush=True)
                    self.rate.invalidate()

        self.last_s, self.last_ms, self.last_samples = s, ms, samples

        if not self.sec_init:
            self.sec_init = True
            self.reset_second(s)
        if not self.ms_bucket_init:
            self.ms_bucket_init = True
            self.reset_ms_bucket(s, ms)

        if (s, ms) != (self.ms_bucket_s, self.ms_bucket_ms):
            self._finalize_ms_bucket()
            self.reset_ms_bucket(s, ms)

        if s != self.sec_s:
            self._print_second_summary()
            self.reset_second(s)

        # Parse IQ: int16 BE, interleaved I/Q
        arr = np.frombuffer(iq_bytes, dtype=">i2",
                            count=2 * K_COMPLEX_PER_PACKET)
        i32 = arr[0::2].astype(np.int32)
        q32 = arr[1::2].astype(np.int32)

        p = (i32 * i32 + q32 * q32).astype(np.float64)
        pkt_sum_p = float(p.sum())
        pkt_max_p = float(p.max())

        rail = int(np.count_nonzero((i32 == 32767) | (i32 == -32768))) + \
               int(np.count_nonzero((q32 == 32767) | (q32 == -32768)))
        near = int(np.count_nonzero((i32 >= NEAR_RAIL) | (i32 <= -NEAR_RAIL))) + \
               int(np.count_nonzero((q32 >= NEAR_RAIL) | (q32 <= -NEAR_RAIL)))
        self.sec_rail += rail
        self.sec_near += near

        self.sec_sum_p += pkt_sum_p
        self.sec_nsamp += K_COMPLEX_PER_PACKET
        if pkt_max_p > self.sec_max_p:
            self.sec_max_p = pkt_max_p

        if not self.args.no_emul:
            i12 = round_shift_right_4(i32)
            q12 = round_shift_right_4(q32)
            i13 = i12 << 1
            q13 = q12 << 1
            p_e = (i13 * i13 + q13 * q13).astype(np.float64)
            pkt_sum_p_e = float(p_e.sum())
            pkt_max_p_e = float(p_e.max())
            self.sec_sum_p_e += pkt_sum_p_e
            self.sec_nsamp_e += K_COMPLEX_PER_PACKET
            if pkt_max_p_e > self.sec_max_p_e:
                self.sec_max_p_e = pkt_max_p_e
        else:
            pkt_sum_p_e = 0.0
            p_e = None

        # CCDF vs running mean (matches main.cpp)
        mean_p_run = self.sec_sum_p / self.sec_nsamp
        if mean_p_run > 0:
            for i, r in enumerate(CCDF_RATIOS):
                self.sec_ccdf[i] += int(np.count_nonzero(p > mean_p_run * r))
        if not self.args.no_emul and self.sec_nsamp_e > 0:
            mean_p_e_run = self.sec_sum_p_e / self.sec_nsamp_e
            if mean_p_e_run > 0:
                for i, r in enumerate(CCDF_RATIOS):
                    self.sec_ccdf_e[i] += int(np.count_nonzero(p_e > mean_p_e_run * r))

        self.ms_sum_p += pkt_sum_p
        self.ms_nsamp += K_COMPLEX_PER_PACKET
        if not self.args.no_emul:
            self.ms_sum_p_e += pkt_sum_p_e
            self.ms_nsamp_e += K_COMPLEX_PER_PACKET

        self.sec_packets_received += 1

        # Feed spectrum (skip NULL packets — they're zero-padded markers)
        if self.spectrum is not None and self.spectrum.fs is not None and not null_flag:
            iq_c = (i32.astype(np.float32) +
                    1j * q32.astype(np.float32)).astype(np.complex64)
            iq_c *= np.float32(1.0 / FULL_SCALE)
            self.spectrum.push(iq_c)


# ---------------- Socket / packet parse ----------------

def make_socket(args) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SO_RCVBUF_BYTES)
    except OSError:
        pass
    actual = s.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    print(f"SO_RCVBUF = {actual} bytes (kernel may cap to net.core.rmem_max)",
          flush=True)
    s.bind((args.bind, args.port))
    if args.mcast_group:
        print(f"Joining multicast group {args.mcast_group}", flush=True)
        mreq = struct.pack("4s4s",
                           socket.inet_aton(args.mcast_group),
                           socket.inet_aton("0.0.0.0"))
        s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    s.settimeout(0.02)
    return s


def parse_packet(data: bytes):
    if len(data) < 8 + 4 * K_COMPLEX_PER_PACKET:
        return None
    b0 = data[0]
    null_flag = bool(b0 & 0x04)
    ms = ((b0 & 0x03) << 8) | data[1]
    samples = (data[2] << 8) | data[3]
    s = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7]
    iq = data[8:8 + 4 * K_COMPLEX_PER_PACKET]
    return s, ms, samples, null_flag, iq


# ---------------- Plot ----------------

def setup_plot(args, metrics: Metrics):
    if args.no_spectrum:
        fig, ax_top = plt.subplots(figsize=(11, 5))
        ax_bot = None
    else:
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(11, 8),
            gridspec_kw={"height_ratios": [1, 1.2]},
        )
    fig.canvas.manager.set_window_title("UDP IQ receiver")

    (line_raw,) = ax_top.plot([], [], lw=1.0, label="RAW per-ms P(dBFS)")
    if args.no_emul:
        line_emul = None
    else:
        (line_emul,) = ax_top.plot([], [], lw=1.0,
                                   label="EMUL(12b+6dB) per-ms P(dBFS)")
    ax_top.set_title("Per-ms power (dBFS)")
    ax_top.set_xlabel(f"Time (s, last {args.history:.0f}s)")
    ax_top.set_ylabel("Power (dBFS)")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="upper right")

    text_top = ax_top.text(
        0.01, 0.02, "", transform=ax_top.transAxes,
        va="bottom", ha="left", fontsize=9, family="monospace",
    )

    if ax_bot is not None:
        (line_min,) = ax_bot.plot([], [], color="gray", lw=0.6, label="min")
        (line_avg,) = ax_bot.plot([], [], color="deepskyblue", lw=1.0, label="avg")
        (line_max,) = ax_bot.plot([], [], color="orange", lw=0.8, label="max")
        ax_bot.set_xlabel("Frequency [MHz]")
        ax_bot.set_ylabel(f"Power [dB / {args.rbw/1e3:.0f} kHz]")
        ax_bot.grid(True, alpha=0.3)
        ax_bot.legend(loc="upper right")
        text_bot = ax_bot.text(
            0.01, 0.02, "waiting for sample-rate detection...",
            transform=ax_bot.transAxes, va="bottom", ha="left",
            fontsize=9, family="monospace",
        )
    else:
        line_min = line_avg = line_max = text_bot = None

    fig.tight_layout()

    def on_key(event):
        if event.key == "c" and metrics.spectrum is not None:
            metrics.spectrum.clear_hold()
            print("Spectrum hold cleared", flush=True)

    fig.canvas.mpl_connect("key_press_event", on_key)

    return {
        "fig": fig, "ax_top": ax_top, "ax_bot": ax_bot,
        "line_raw": line_raw, "line_emul": line_emul,
        "line_min": line_min, "line_avg": line_avg, "line_max": line_max,
        "text_top": text_top, "text_bot": text_bot,
    }


def update_top(args, metrics, w):
    if len(metrics.plot_t) < 2:
        return
    t_now = time.time()
    t0 = t_now - args.history
    tt = np.fromiter(metrics.plot_t, dtype=np.float64, count=len(metrics.plot_t))
    yr = np.fromiter(metrics.plot_raw, dtype=np.float64, count=len(metrics.plot_raw))
    x = tt - t0
    m = x >= 0.0
    xw = x[m]
    yrw = yr[m]
    w["line_raw"].set_data(xw, yrw)

    if w["line_emul"] is not None and len(metrics.plot_emul) == len(metrics.plot_raw):
        ye = np.fromiter(metrics.plot_emul, dtype=np.float64,
                         count=len(metrics.plot_emul))
        yew = ye[m]
        w["line_emul"].set_data(xw, yew)
        finite_e = np.isfinite(yew)
    else:
        yew = np.array([])
        finite_e = np.array([], dtype=bool)

    w["ax_top"].set_xlim(0.0, args.history)
    finite_r = np.isfinite(yrw)
    if finite_r.any() or finite_e.any():
        vals = np.concatenate([yrw[finite_r], yew[finite_e]])
        ymin = float(vals.min()) - 1.0
        ymax = float(vals.max()) + 1.0
        if ymax - ymin < 5.0:
            mid = 0.5 * (ymin + ymax)
            ymin, ymax = mid - 2.5, mid + 2.5
        w["ax_top"].set_ylim(ymin, ymax)

    if metrics.sec_nsamp > 0:
        rms = dbfs_rms_from_meanp(metrics.sec_sum_p / metrics.sec_nsamp)
        std = metrics.ms_stats.stddev()
        p2p = metrics.ms_stats.p2p()
        ccdf = [c / metrics.sec_nsamp for c in metrics.sec_ccdf]
        w["text_top"].set_text(
            f"s={metrics.sec_s}  RMS={rms:6.2f} dBFS\n"
            f"per-ms std={std:.3f} dB  p2p={p2p:.3f} dB\n"
            f"CCDF(+8/+10/+12)={ccdf[0]:.6f}/{ccdf[1]:.6f}/{ccdf[2]:.6f}"
        )
    else:
        w["text_top"].set_text("waiting for packets...")


def update_bot(args, metrics, w):
    if w["ax_bot"] is None or metrics.spectrum is None:
        return
    metrics.spectrum.maybe_update(time.time())
    triple = metrics.spectrum.get_min_avg_max()
    if triple is None:
        return
    pmin, pavg, pmax = triple
    f = metrics.spectrum.f
    f_mhz = (f + args.fc) / 1e6
    w["line_min"].set_data(f_mhz, pmin)
    w["line_avg"].set_data(f_mhz, pavg)
    w["line_max"].set_data(f_mhz, pmax)

    w["ax_bot"].set_xlim(f_mhz[0], f_mhz[-1])
    vals = np.concatenate([pmin, pmax])
    finite = np.isfinite(vals)
    if finite.any():
        ymin = float(vals[finite].min()) - 3.0
        ymax = float(vals[finite].max()) + 3.0
        w["ax_bot"].set_ylim(ymin, ymax)

    fs = metrics.spectrum.fs
    n = metrics.spectrum.n_traces()
    w["text_bot"].set_text(
        f"fs={fs/1e6:.3f} MHz  RBW={args.rbw/1e3:.0f} kHz  "
        f"hold={args.hold:.1f}s ({n} traces)  press 'c' to clear"
    )


# ---------------- Main loop ----------------

def main():
    args = parse_args()
    sock = make_socket(args)
    metrics = Metrics(args)
    widgets = setup_plot(args, metrics)

    def update(_frame):
        # Drain socket for a short budget per animation tick.
        t_start = time.time()
        while (time.time() - t_start) < 0.02:
            try:
                data, _addr = sock.recvfrom(BUFFERSIZE)
            except socket.timeout:
                break
            except OSError as e:
                print(f"socket error: {e}", flush=True)
                break
            pkt = parse_packet(data)
            if pkt is None:
                continue
            metrics.process_packet(*pkt)

        update_top(args, metrics, widgets)
        update_bot(args, metrics, widgets)

        artists = [widgets["line_raw"], widgets["text_top"]]
        if widgets["line_emul"] is not None:
            artists.append(widgets["line_emul"])
        if widgets["ax_bot"] is not None:
            artists.extend([widgets["line_min"], widgets["line_avg"],
                            widgets["line_max"], widgets["text_bot"]])
        return artists

    _ani = FuncAnimation(widgets["fig"], update, interval=50, blit=False,
                         cache_frame_data=False)
    try:
        plt.show()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
