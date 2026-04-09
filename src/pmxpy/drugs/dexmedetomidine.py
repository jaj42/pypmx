class Hannivoort:
    def __init__(self, weight: int):
        """3 compartment dexmedetomidine Pk model

        Units:
        weight(kg)

        Reference:
        Hannivoort, L, et al
        Development of an Optimized Pharmacokinetic Model of Dexmedetomidine Using Target-controlled Infusion in Healthy Volunteers
        Anesthesiology 8 2015, Vol.123, 357-367.
        doi:10.1097/ALN.0000000000000740
        """

        self.v1 = 1.78 * (weight / 70)
        self.v2 = 30.3 * (weight / 70)
        self.v3 = 52.0 * (weight / 70)

        self.Q1 = 0.686 * (weight / 70) ** 0.75
        self.Q2 = 2.98 * (self.v2 / 30.3) ** 0.75
        self.Q3 = 0.602 * (self.v3 / 52.0) ** 0.75

        self.from_clearances()

        self.keo = 0


class Dyck:
    def __init__(self, height: int):
        """

        Units:
        Height(cm)

        Reference:
        Dyck, JB, et al
        Computer-controlled infusion of intravenous dexmedetomidine hydrochloride in adult human volunteers
        Anesthesiology. 1993 May;78(5):821-8.
        PMID:8098191 DOI:10.1097/00000542-199305000-00003
        """

        self.v1 = 7.99
        self.v2 = 13.8
        self.v3 = 187

        self.Q1 = round((0.00791 * height) - 0.928, 4)
        self.Q2 = 2.26
        self.Q3 = 1.99

        self.from_clearances()

        self.keo = 0


# class Dexmedetomidine:
#    def __init__(
#        self, weight: int, t0: datetime | None = None, y0: List[float] = [0, 0, 0, 0]
#    ) -> None:
#        self._dt = t0 if t0 is not None else datetime.now()
#        self._y = y0
#        self._rate = 0
#        self.model = Hannivoort(weight)
#        self._params = {
#            "V1": self.model.v1,
#            "V2": self.model.v2,
#            "V3": self.model.v3,
#            "k12": self.model.k12,
#            "k21": self.model.k21,
#            "k13": self.model.k13,
#            "k31": self.model.k31,
#            "k10": self.model.k10,
#            # Colin PJ, Hannivoort LN, Eleveld DJ, Reyntjens KMEM, Absalom AR, Vereecke HEM, Struys MMRF.
#            # Dexmedetomidine pharmacokinetic-pharmacodynamic modelling in healthy volunteers:
#            # 1. Influence of arousal on bispectral index and sedation.
#            # Br J Anaesth. 2017 Aug 1;119(2):200-210. doi: 10.1093/bja/aex085. PMID: 28854538.
#            "ke0": 0.120 / 60,
#            "C50": 2.63,
#            "BaseBIS": 96.8,
#        }
#
#    @property
#    def rate(self):
#        return self._rate
#
#    @rate.setter
#    def rate(self, value):
#        self.run_pk()
#        self._rate = float(value)
#
#    def give_bolus(self, dose):
#        self._y[0] += dose / self._params["V1"]
#
#    def pk(self, t, y, Inf, p):
#        C1, C2, C3, Ce = y
#        ddt_C1 = (
#            Inf
#            + C2 * p["V2"] * p["k21"]
#            + C3 * p["V3"] * p["k31"]
#            - C1 * p["V1"] * (p["k12"] + p["k13"] + p["k10"])
#        ) / p["V1"]
#        ddt_C2 = (C1 * p["V1"] * p["k12"] - C2 * p["V2"] * p["k21"]) / p["V2"]
#        ddt_C3 = (C1 * p["V1"] * p["k13"] - C3 * p["V3"] * p["k31"]) / p["V3"]
#        ddt_Ce = p["ke0"] * (C1 - Ce)
#        return ddt_C1, ddt_C2, ddt_C3, ddt_Ce
#
#    @property
#    def cc(self):
#        return self._y[0]
#
#    @property
#    def ce(self):
#        return self._y[3]
#
#    def calc_effect(self, Ce=None):
#        if Ce is None:
#            Ce = self.ce
#        return self._params["BaseBIS"] * (1 - Ce / (Ce + self._params["C50"]))
#
#    def run_pkpd(self, dt=None):
#        if dt is None:
#            dt = datetime.now()
#        t = (dt - self._dt).total_seconds()
#        res = solve_ivp(
#            self.pk,
#            t_span=[0, t],
#            t_eval=[t],
#            y0=self._y,
#            args=(self._rate, self._params),
#        )
#        self._dt = dt
#        ytmp = np.squeeze(res.y)
#        if len(ytmp) == len(self._y):
#            self._y = ytmp
#        else:
#            # TODO: log error
#            pass
#        return dt, self.cc, self.calc_effect()
#
#    def simulate_pkpd(self, dt, bolus=None, inf_rate=None, npoints=100):
#        if bolus is None and inf_rate is None:
#            # Reset
#            return ([], [], [])
#        t0 = self._dt
#        y0 = self._y.copy()
#        t = (dt - t0).total_seconds()
#        t_span = [0, t]
#        if bolus:
#            y0[0] += bolus / self._params["V1"]
#        rate = inf_rate if inf_rate is not None else self._rate
#        sol_pk = solve_ivp(
#            self.pk, t_span=t_span, y0=y0, args=(rate, self._params), dense_output=True
#        )
#
#        t = np.linspace(*t_span, npoints)
#        z = sol_pk.sol(t)
#        Cc = z[0]
#        Ce = z[3]
#
#        calc_effect_v = np.vectorize(self.calc_effect)
#        E = calc_effect_v(Ce)
#
#        t_out = [t0 + timedelta(seconds=_t) for _t in t]
#
#        return (t_out, Cc, E)
