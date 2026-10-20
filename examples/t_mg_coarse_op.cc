
#include "qdp.h"

#include <cmath>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace QDP;


// ---------------------------------------------------------------------
// ---------------------------------------------------------------------

using CoarseGaugeOffd = multi1d< LatticePropagator >;
using CoarseGaugeDiag = LatticePropagator;

struct CoarseGauge
{
  CoarseGauge(): offd(Nd) {}
  CoarseGauge(int nd): offd(nd) {}

  CoarseGaugeOffd offd;
  CoarseGaugeDiag diag;
};

struct CoarseEvenOddGauge
{
  CoarseEvenOddGauge():H_eo( 2 * Nd ) {}

  CoarseGauge     H_eo;      // 2*Nd offd; H_oe is derived from it
  CoarseGaugeDiag A_ee;
  CoarseGaugeDiag A_ee_inv;
  CoarseGaugeDiag A_oo;
};


inline LatticeFermion coarse_g5(const LatticeFermion& psi)
{
  assert( psi.shape().ns() == 2 );

  LatticeFermion ret = psi;
  pokeSpin( ret, -peekSpin(psi, 1) , 1 );
  return ret;
}


enum class Stencil { Separate, Fused, Preshifted };

inline const char* stencil_name(Stencil s)
{
  switch (s)
    {
    case Stencil::Separate:   return "separate";
    case Stencil::Fused:      return "fused";
    case Stencil::Preshifted: return "fused, pre-shifted links";
    }
  return "";
}


template<typename T>
class CoarseWilsonDslash
{
  CoarseGauge g;
  CoarseGauge gs;
  Stencil     stencil;

public:
  CoarseWilsonDslash( const CoarseGauge& g_, Stencil stencil_ ): g(g_), gs(stencil_ == Stencil::Preshifted ? 2 * Nd : 0), stencil(stencil_)
  {
    assert( g.offd.size() == 2 * Nd );

    if (stencil == Stencil::Preshifted)
      for (int mu = 0; mu < Nd; ++mu)
	{
	  gs.offd[mu]      = shift(g.offd[mu],      BACKWARD, mu);
	  gs.offd[Nd + mu] = shift(g.offd[Nd + mu], FORWARD,  mu);
	}
  }

  //! chi = M psi  on checkerboard cb
  void apply(T& chi, const T& psi, int cb) const
  {
    if (stencil == Stencil::Preshifted)
      {
	chi[rb[cb]] =
	  shift(gs.offd[0] * psi, FORWARD, 0) +
	  shift(gs.offd[1] * psi, FORWARD, 1) +
	  shift(gs.offd[2] * psi, FORWARD, 2) +
	  shift(gs.offd[3] * psi, FORWARD, 3) +
	  shift(gs.offd[4] * psi, BACKWARD, 0) +
	  shift(gs.offd[5] * psi, BACKWARD, 1) +
	  shift(gs.offd[6] * psi, BACKWARD, 2) +
	  shift(gs.offd[7] * psi, BACKWARD, 3) +
	  g.diag * psi;
	return;
      }

    if (stencil == Stencil::Fused)
      {
	chi[rb[cb]] =
	  g.offd[0] * shift(psi, FORWARD, 0) +
	  g.offd[1] * shift(psi, FORWARD, 1) +
	  g.offd[2] * shift(psi, FORWARD, 2) +
	  g.offd[3] * shift(psi, FORWARD, 3) +
	  g.offd[4] * shift(psi, BACKWARD, 0) +
	  g.offd[5] * shift(psi, BACKWARD, 1) +
	  g.offd[6] * shift(psi, BACKWARD, 2) +
	  g.offd[7] * shift(psi, BACKWARD, 3) +
	  g.diag * psi;
	return;
      }

    chi[rb[cb]]  = g.offd[0] * shift(psi, FORWARD, 0);
    chi[rb[cb]] += g.offd[1] * shift(psi, FORWARD, 1);
    chi[rb[cb]] += g.offd[2] * shift(psi, FORWARD, 2);
    chi[rb[cb]] += g.offd[3] * shift(psi, FORWARD, 3);
    chi[rb[cb]] += g.offd[4] * shift(psi, BACKWARD, 0);
    chi[rb[cb]] += g.offd[5] * shift(psi, BACKWARD, 1);
    chi[rb[cb]] += g.offd[6] * shift(psi, BACKWARD, 2);
    chi[rb[cb]] += g.offd[7] * shift(psi, BACKWARD, 3);
    chi[rb[cb]] += g.diag * psi;
  }

  //! chi = M^dag psi  on checkerboard cb
  void applyDagger(T& chi, const T& psi, int cb) const
  {
    // transposed stencil: same links, adjointed, shifts swapped
    if (stencil != Stencil::Separate)
      {
	chi[rb[cb]] =
	  adj(g.diag) * psi +
	  shift(adj(g.offd[4]) * psi, FORWARD,  0) +
	  shift(adj(g.offd[5]) * psi, FORWARD,  1) +
	  shift(adj(g.offd[6]) * psi, FORWARD,  2) +
	  shift(adj(g.offd[7]) * psi, FORWARD,  3) +
	  shift(adj(g.offd[0]) * psi, BACKWARD, 0) +
	  shift(adj(g.offd[1]) * psi, BACKWARD, 1) +
	  shift(adj(g.offd[2]) * psi, BACKWARD, 2) +
	  shift(adj(g.offd[3]) * psi, BACKWARD, 3);
	return;
      }

    chi[rb[cb]]  = adj(g.diag) * psi;
    chi[rb[cb]] += shift(adj(g.offd[4]) * psi, FORWARD,  0);
    chi[rb[cb]] += shift(adj(g.offd[5]) * psi, FORWARD,  1);
    chi[rb[cb]] += shift(adj(g.offd[6]) * psi, FORWARD,  2);
    chi[rb[cb]] += shift(adj(g.offd[7]) * psi, FORWARD,  3);
    chi[rb[cb]] += shift(adj(g.offd[0]) * psi, BACKWARD, 0);
    chi[rb[cb]] += shift(adj(g.offd[1]) * psi, BACKWARD, 1);
    chi[rb[cb]] += shift(adj(g.offd[2]) * psi, BACKWARD, 2);
    chi[rb[cb]] += shift(adj(g.offd[3]) * psi, BACKWARD, 3);
  }

  //! chi = ghat5 M^dag ghat5 psi   (= H_oe when these are the H_eo links)
  void applyG5Dagger(T& chi, const T& psi, int cb) const
  {
    T t = coarse_g5(psi);
    T u = zero;
    applyDagger(u, t, cb);
    chi[rb[cb]] = coarse_g5(u);
  }

  const CoarseGauge& gauge() const { return g; }
};


template<typename T>
class CoarseEvenOddSchur
{
  CoarseEvenOddGauge    g;
  CoarseWilsonDslash<T> Heo;

public:
  CoarseEvenOddSchur(const CoarseEvenOddGauge& g_, Stencil stencil) : g(g_), Heo(g_.H_eo, stencil) {}

  void applyHeo(T& chi, const T& psi) const
  {
    Heo.apply(chi, psi, 0);
    Heo.apply(chi, psi, 1);
  }

  //! H_oe = ghat5 H_eo^dag ghat5   -- no stored links
  void applyHoe(T& chi, const T& psi) const
  {
    Heo.applyG5Dagger(chi, psi, 0);
    Heo.applyG5Dagger(chi, psi, 1);
  }

  void schur(T& chi, const T& psi) const
  {
    T t1 = zero;
    T t2 = zero;

    applyHeo(t1, psi);              // t1 = H_eo psi
    t2 = g.A_ee_inv * t1;           // t2 = (A_ee^c)^{-1} t1
    applyHoe(t1, t2);               // t1 = H_oe t2  (derived, no links)
    chi = g.A_oo * psi - t1;
  }

  const CoarseEvenOddGauge&    gauge()  const { return g; }
  const CoarseWilsonDslash<T>& dslash() const { return Heo; }
};

// ---------------------------------------------------------------------
// ---------------------------------------------------------------------


namespace
{
  struct Config
  {
    std::string   name;
    JitDataFormat links;
    JitDataFormat vectors;
    Stencil       stencil;
    bool          round_half = false;
    std::string   compare_to = "qdp-mg";
  };

  std::string data_format_name(JitDataFormat fmt)
  {
    std::string s = fmt.order == JitDataOrder::SiteFast ? "SiteFast" : "SiteSlow";
    return fmt.half() ? s + " half" : s;
  }


  template<class T>
  double bytes_of(const OLattice<T>& a)
  {
    double site = WordSize<T>::value(a.shape()) * sizeof(typename WordType<T>::Type_t);
    return site * a.layout().sitesOnNode();
  }


  long gpu_launches()
  {
    std::set<JitFunction::Func_t> seen;
    long n = 0;
    for (JitFunction* f : gpu_get_functions())
      if (!f->empty() && f->isGPU() && seen.insert(f->get_function()).second)
	n += f->get_call_counter();
    return n;
  }


  void bench(const std::string& name, int iter, double bytes, const std::function<void()>& f)
  {
    f();

    long l0 = gpu_launches();

    StopWatch w;
    w.start();
    for (int i = 0; i < iter; ++i)
      f();
    w.stop();

    double us_per_call = w.getTimeInMicroseconds() / iter;
    double launches = double(gpu_launches() - l0) / iter;
    double gbs = bytes / (us_per_call * 1.0e-6) * 1.0e-9;

    char buf[256];
    snprintf(buf, sizeof buf, "  %-34s %10.1f us/call  %6.1f kernels/call  %7.1f GB/s (effective)",
	     name.c_str(), us_per_call, launches, gbs);
    QDPIO::cout << buf << std::endl;
  }


  const int n_probes  = 4;
  const int per_level = 1 + 3 * n_probes;

  std::vector<double> checksums(const LatticeFermion& chi_in)
  {
    LatticeFermion chi(JitDataOrder::SiteFast);
    chi = chi_in;

    std::vector<double> c;
    c.push_back(toDouble(norm2(chi)));

    for (int k = 0; k < n_probes; ++k)
      {
	LatticeFermion probe(JitDataOrder::SiteFast);
	gaussian(probe);
	DComplex ip = innerProduct(probe, chi);
	c.push_back(toDouble(norm2(probe)));
	c.push_back(toDouble(real(ip)));
	c.push_back(toDouble(imag(ip)));
      }
    return c;
  }


  double compare(const std::vector<double>& ref, const std::vector<double>& sums)
  {
    if (ref.size() != sums.size())
      return 1.0e30;

    double worst = 0;
    for (size_t i = 0; i < sums.size(); ++i)
      {
	const size_t base = (i / per_level) * per_level;
	const size_t j = i - base;
	double scale;
	if (j == 0 || (j - 1) % 3 == 0)
	  scale = std::fabs(ref[i]);
	else
	  scale = std::sqrt(ref[base] * ref[base + 1 + 3 * ((j - 1) / 3)]);
	worst = std::max(worst, std::fabs(sums[i] - ref[i]) / scale);
      }
    return worst;
  }

  const double tolerance = 1e-6;


  std::vector<double> run_config(const Config& cfg, int iter,
				 const CoarseEvenOddGauge& g0, const LatticeFermion& psi0)
  {
    QDPIO::cout << "\n [" << cfg.name << "] links " << data_format_name(cfg.links) << ", vectors " << data_format_name(cfg.vectors)
		<< ", stencil " << stencil_name(cfg.stencil) << std::endl;

    auto copy_gauge = [](CoarseEvenOddGauge& d, const CoarseEvenOddGauge& s) {
      for (int mu = 0; mu < 2 * Nd; ++mu)
	d.H_eo.offd[mu] = s.H_eo.offd[mu];
      d.H_eo.diag = s.H_eo.diag;
      d.A_ee      = s.A_ee;
      d.A_ee_inv  = s.A_ee_inv;
      d.A_oo      = s.A_oo;
    };

    std::unique_ptr< CoarseEvenOddSchur<LatticeFermion> > op;
    {
      std::unique_ptr<CoarseEvenOddGauge> gh;
      if (cfg.round_half)
	{
	  JitDataFormatScope scope(JitDataFormat(JitDataOrder::SiteFast, JitStorage::Half));
	  gh.reset(new CoarseEvenOddGauge);
	  copy_gauge(*gh, g0);
	}

      JitDataFormatScope scope(cfg.links);
      CoarseEvenOddGauge g;
      copy_gauge(g, cfg.round_half ? *gh : g0);
      op.reset(new CoarseEvenOddSchur<LatticeFermion>(g, cfg.stencil));
    }
    const CoarseEvenOddGauge& g = op->gauge();

    JitDataFormatScope scope(cfg.vectors);

    LatticeFermion psi, chi, t1, t2;
    psi = psi0;
    chi = zero;
    t1 = zero;
    t2 = zero;

    if (g.A_oo.data_format() != cfg.links ||
	op->dslash().gauge().offd[0].data_format() != cfg.links ||
	psi.data_format() != cfg.vectors)
      {
	QDPIO::cerr << "data order not as requested" << std::endl;
	QDP_abort(1);
      }

    const double F = bytes_of(psi);
    const double U = bytes_of(g.A_oo);

    bench("schur", iter, 20 * U + 40 * F, [&]() { op->schur(chi, psi); });

    bench("H_eo (both cb)", iter, 9 * U + 18 * F, [&]() { op->applyHeo(t1, psi); });
    bench("H_oe (both cb)", iter, 9 * U + 18 * F, [&]() { op->applyHoe(t1, psi); });

    if (cfg.stencil == Stencil::Separate)
      {
	bench("t2 = A_ee_inv * t1", iter, U + 2 * F, [&]() { t2 = g.A_ee_inv * t1; });
	bench("chi = A_oo * psi - t1", iter, U + 3 * F, [&]() { chi = g.A_oo * psi - t1; });
	bench("coarse_g5", iter, 2 * F, [&]() { t2 = coarse_g5(t1); });

	const multi1d<LatticePropagator>& offd = op->dslash().gauge().offd;
	int k = 0;
	auto link = [&]() -> const LatticePropagator& { return offd[k++ % (2 * Nd)]; };

	bench("chi[rb0]  = U * shift(psi,+0)", iter, (U + 2 * F) / 2,
	      [&]() { chi[rb[0]] = link() * shift(psi, FORWARD, 0); });
	bench("chi[rb0] += U * shift(psi,+0)", iter, (U + 3 * F) / 2,
	      [&]() { chi[rb[0]] += link() * shift(psi, FORWARD, 0); });
	bench("chi[rb0] += shift(adj(U)*psi,-0)", iter, (U + 3 * F) / 2,
	      [&]() { chi[rb[0]] += shift(adj(link()) * psi, BACKWARD, 0); });
	bench("chi[rb0] += U * psi", iter, (U + 3 * F) / 2,
	      [&]() { chi[rb[0]] += link() * psi; });
      }

    op->schur(chi, psi);
    return checksums(chi);
  }


  bool run_level(int level, int iter, const std::vector<Config>& configs, std::vector<double>& sums)
  {
    CoarseEvenOddGauge g0;
    LatticeFermion psi0;

    const int ns = psi0.shape().ns();
    const int nc = psi0.shape().nc();

    multi1d<int> lat = Layout::current().lattSize();
    QDPIO::cout << "\nLevel " << level << ": " << lat[0] << "x" << lat[1] << "x" << lat[2] << "x" << lat[3]
		<< ", Ns = " << ns << ", Nc = " << nc
		<< ", sites on node = " << Layout::current().sitesOnNode() << std::endl;

    for (int mu = 0; mu < 2 * Nd; ++mu)
      gaussian(g0.H_eo.offd[mu]);
    gaussian(g0.H_eo.diag);
    gaussian(g0.A_ee);
    gaussian(g0.A_ee_inv);
    gaussian(g0.A_oo);
    gaussian(psi0);

    Seed probe_seed;
    RNG::savern(probe_seed);

    bool ok = true;
    std::vector<double> ref;
    std::map<std::string, std::vector<double> > by_name;

    for (size_t i = 0; i < configs.size(); ++i)
      {
	RNG::setrn(probe_seed);
	std::vector<double> c = run_config(configs[i], iter, g0, psi0);

	if (i == 0)
	  {
	    ref = c;
	    QDPIO::cout << "  checksums:";
	    for (double x : c)
	      QDPIO::cout << " " << x;
	    QDPIO::cout << std::endl;
	  }
	else if (configs[i].compare_to.empty())
	  {
	    QDPIO::cout << "  checksums vs qdp-mg configuration: relative difference "
			<< compare(ref, c) << " (not checked)" << std::endl;
	  }
	else
	  {
	    double d = compare(by_name.at(configs[i].compare_to), c);
	    QDPIO::cout << "  checksums vs " << configs[i].compare_to << " configuration: "
			<< (d < tolerance ? "OK" : "MISMATCH")
			<< " (largest relative difference " << d << ")" << std::endl;
	    ok = ok && d < tolerance;
	  }
	by_name[configs[i].name] = c;
      }

    sums.insert(sums.end(), ref.begin(), ref.end());
    return ok;
  }
}


int main(int argc, char **argv)
{
  QDP_initialize(&argc, &argv);

  multi1d<int> nrow(Nd);
  nrow[0] = 16; nrow[1] = 16; nrow[2] = 16; nrow[3] = 48;
  int iter = 200;
  int levels = 2;
  std::string write_ref, check_ref;

  for (int i = 1; i < argc; i++)
    {
      if (strcmp(argv[i], "-nrow") == 0)
	for (int mu = 0; mu < Nd; ++mu)
	  sscanf(argv[++i], "%d", &nrow[mu]);
      else if (strcmp(argv[i], "-iter") == 0)
	sscanf(argv[++i], "%d", &iter);
      else if (strcmp(argv[i], "-levels") == 0)
	sscanf(argv[++i], "%d", &levels);
      else if (strcmp(argv[i], "-write-ref") == 0)
	write_ref = argv[++i];
      else if (strcmp(argv[i], "-check-ref") == 0)
	check_ref = argv[++i];
    }

  Layout::setLattSize(nrow);
  Layout::create();

  Seed seed = 11;
  RNG::setrn(seed);

  QDPIO::cout << "fine lattice = " << nrow[0] << "x" << nrow[1] << "x" << nrow[2] << "x" << nrow[3]
	      << ",  iterations per measurement = " << iter << std::endl;

  const JitDataFormat Fast = JitDataOrder::SiteFast;
  const JitDataFormat Slow = JitDataOrder::SiteSlow;
  const JitDataFormat FastHalf(JitDataOrder::SiteFast, JitStorage::Half);
  const JitDataFormat SlowHalf(JitDataOrder::SiteSlow, JitStorage::Half);

  const Stencil Sep = Stencil::Separate;
  const Stencil Fus = Stencil::Fused;
  const Stencil Pre = Stencil::Preshifted;

  const std::vector<Config> configs = {
    { "qdp-mg",                        Fast, Fast, Sep },
    { "fused",                         Fast, Fast, Fus },
    { "fused, pre-shifted",            Fast, Fast, Pre },
    { "slow links",                    Slow, Fast, Sep },
    { "slow links, fused",             Slow, Fast, Fus },
    { "slow links, fused, pre-shifted",Slow, Fast, Pre },
    { "slow all",                      Slow, Slow, Sep },
    { "slow all, fused",               Slow, Slow, Fus },
    { "half-rounded",                  Fast,     Fast, Sep, true, "" },
    { "half links",                    FastHalf, Fast, Sep, false, "half-rounded" },
    { "half links, fused",             FastHalf, Fast, Fus, false, "half-rounded" },
    { "half links, slow",              SlowHalf, Fast, Sep, false, "half-rounded" },
  };

  multi1d<int> blocking(Nd);
  blocking = 2;

  bool ok = true;
  std::vector<double> sums;
  {
    std::vector< std::unique_ptr<LayoutShapeScope> > grids;
    for (int level = 1; level <= levels; ++level)
      {
	grids.emplace_back(new LayoutShapeScope(Layout::current().coarsen(blocking), 2, 12));
	ok = run_level(level, iter, configs, sums) && ok;
      }
    while (!grids.empty())
      grids.pop_back();
  }

  if (!write_ref.empty() && Layout::primaryNode())
    {
      std::ofstream f(write_ref);
      f.precision(17);
      for (double x : sums)
	f << x << "\n";
      QDPIO::cout << "\nchecksums written to " << write_ref << std::endl;
    }

  if (!check_ref.empty())
    {
      std::ifstream f(check_ref);
      std::vector<double> ref;
      double x;
      while (f >> x)
	ref.push_back(x);

      double d = compare(ref, sums);
      QDPIO::cout << "\nchecksums vs " << check_ref << ": " << (d < tolerance ? "OK" : "MISMATCH")
		  << " (largest relative difference " << d << ")" << std::endl;
      ok = ok && d < tolerance;
    }

  QDP_finalize();
  return ok ? 0 : 1;
}
