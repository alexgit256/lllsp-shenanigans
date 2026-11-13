#ifndef LLLSP_SANDPILE_QD_CORE_HPP
#define LLLSP_SANDPILE_QD_CORE_HPP

#include <qd/dd_real.h>
#include <qd/qd_real.h>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

namespace lllsp {

class QDCore {
public:
  virtual ~QDCore() = default;
  virtual void set_r_dd(std::size_t i, const dd_real& v) = 0;
  virtual void set_mu_dd(std::size_t i, std::size_t j, const dd_real& v) = 0;
  virtual void set_r_qd(std::size_t i, const qd_real& v) = 0;
  virtual void set_mu_qd(std::size_t i, std::size_t j, const qd_real& v) = 0;
  virtual void set_r_string(std::size_t i, const std::string& v) = 0;
  virtual void set_mu_string(std::size_t i, std::size_t j, const std::string& v) = 0;
  virtual std::string r_string(std::size_t i) const = 0;
  virtual std::string mu_string(std::size_t i, std::size_t j) const = 0;
  virtual void run(std::uint64_t max_swaps, std::size_t start,
                             std::size_t stop, bool trace,
                             std::vector<std::string>& trace_out,
                             std::uint64_t& tests, std::uint64_t& swaps) = 0;
  virtual void resample_external(std::size_t start, std::size_t stop) = 0;
  virtual void check() const = 0;
};

// A single arithmetic kernel shared by QD's double-double and quad-double types.
// The Cython bridge copies fplll's native FP_NR<T>::get_data() directly into it.
template <class Real> struct NativeCast;
template <> struct NativeCast<dd_real> {
  static dd_real from_dd(const dd_real& x) { return x; }
  static dd_real from_qd(const qd_real& x) {
    return dd_real(x.to_string(75).c_str());
  }
};

template <> struct NativeCast<qd_real> {
  static qd_real from_dd(const dd_real& x) {
    return qd_real(x.to_string(40).c_str());
  }
  static qd_real from_qd(const qd_real& x) { return x; }
};

template <class Real, int DIGITS>
class QDCoreImpl final : public QDCore {
  std::size_t n_;
  std::vector<Real> r_, mu_;
  Real delta_, eta_, siegel_coefficient_;
  bool siegel_;
  std::uint64_t rng_;

  std::uint64_t next64() {
    std::uint64_t x = rng_;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng_ = x;
    return x * UINT64_C(2685821657736338717);
  }

  Real sample() {
    // 106 random bits for dd and 212 bits for qd, using 53-bit chunks.
    Real u(0.0);
    const int chunks = DIGITS == 40 ? 2 : 4;
    const Real radix("1.1102230246251565404236316680908203125e-16"); // 2^-53
    for (int k = 0; k < chunks; ++k) {
      const std::uint64_t bits = next64() >> 11;
      u = (u + Real(static_cast<double>(bits))) * radix;
    }
    return ((u - Real(0.5)) * Real(2.0)) * eta_;
  }

  void sample_mu(std::size_t row, std::size_t col) {
    mu_[row*n_+col] = sample();
  }

 static std::string format(const Real& x) {
  return x.to_string(DIGITS);
}
  static Real parse(const std::string& s) {
    return Real(s.c_str());
  }
  static bool finite(const Real& x) {
    // QD quantities have a binary64 exponent range; get components via to_double.
    return std::isfinite(to_double(x));
  }

public:
  QDCoreImpl(std::size_t n, double delta, double eta, bool siegel,
             std::uint64_t seed)
    : n_(n), r_(n, Real(0.0)), mu_(n*n, Real(0.0)),
      delta_(delta), eta_(eta),
      siegel_coefficient_(delta_ - eta_*eta_), siegel_(siegel),
      rng_(seed ? seed : UINT64_C(0x9e3779b97f4a7c15)) {
    if (n < 2 || !std::isfinite(delta) || !std::isfinite(eta) ||
        !(delta > 0 && delta <= 1) || !(eta > 0) ||
        !(siegel_coefficient_ > Real(0.0)))
      throw std::invalid_argument("invalid dimension, delta or eta");
  }

  void set_r_dd(std::size_t i, const dd_real& v) override { r_.at(i) = NativeCast<Real>::from_dd(v); }
  void set_mu_dd(std::size_t i, std::size_t j, const dd_real& v) override {
    mu_.at(i*n_+j) = NativeCast<Real>::from_dd(v);
  }
  void set_r_qd(std::size_t i, const qd_real& v) override { r_.at(i) = NativeCast<Real>::from_qd(v); }
  void set_mu_qd(std::size_t i, std::size_t j, const qd_real& v) override {
    mu_.at(i*n_+j) = NativeCast<Real>::from_qd(v);
  }
  void set_r_string(std::size_t i, const std::string& v) override { r_.at(i) = parse(v); }
  void set_mu_string(std::size_t i, std::size_t j, const std::string& v) override {
    mu_.at(i*n_+j) = parse(v);
  }
  std::string r_string(std::size_t i) const override { return format(r_.at(i)); }
  std::string mu_string(std::size_t i, std::size_t j) const override {
    return format(mu_.at(i*n_+j));
  }
  void check() const override {
    for (const auto& x : r_) if (!finite(x) || !(x > Real(0.0)))
      throw std::runtime_error("invalid squared GSO diagonal");
    for (const auto& x : mu_) if (!finite(x))
      throw std::runtime_error("nonfinite mu entry");
  }
  void resample_external(std::size_t start, std::size_t stop) override {
    for (std::size_t row=stop; row<n_; ++row)
      for (std::size_t col=start; col<stop; ++col) sample_mu(row, col);
  }
  void run(std::uint64_t max_swaps, std::size_t start,
                    std::size_t stop, bool trace,
                    std::vector<std::string>& trace_out,
                    std::uint64_t& tests, std::uint64_t& swaps) override {
    std::size_t i = start;
    while (i+1 < stop) {
      ++tests;
      const Real mui = mu_[(i+1)*n_+i];
      const Real lhs = (siegel_ ? siegel_coefficient_ : delta_) * r_[i];
      const Real rhs = siegel_ ? r_[i+1] : r_[i+1] + mui*mui*r_[i];
      if (lhs <= rhs) { ++i; continue; }
      if (swaps >= max_swaps)
        throw std::runtime_error("max_swaps reached; state preserved");
      const Real old = r_[i], next = r_[i+1];
      const Real updated = next + mui*mui*old;
      if (!finite(updated) || !(updated > Real(0.0)))
        throw std::runtime_error("bad GSO diagonal after swap");
      const Real new_next = old*next/updated;
      if (!finite(new_next) || !(new_next > Real(0.0)))
        throw std::runtime_error("bad GSO diagonal after swap");
      if (trace) trace_out.push_back(format(mui));
      r_[i] = updated;
      r_[i+1] = new_next;
      ++swaps;
      sample_mu(i+1, i);
      if (i > start) sample_mu(i, i-1);
      if (i+1 < stop-1) sample_mu(i+2, i+1);
      if (i > start) --i;
    }
  }
};

inline QDCore* make_dd(std::size_t n, double delta, double eta,
                       bool siegel, std::uint64_t seed) {
  return new QDCoreImpl<dd_real, 40>(n, delta, eta, siegel, seed);
}
inline QDCore* make_qd(std::size_t n, double delta, double eta,
                       bool siegel, std::uint64_t seed) {
  return new QDCoreImpl<qd_real, 75>(n, delta, eta, siegel, seed);
}

} // namespace lllsp
#endif
