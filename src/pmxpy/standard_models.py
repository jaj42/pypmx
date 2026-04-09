def pk_3comp(t, y, Inf, p):
    A1, A2, A3, Ce = y
    ddt_A1 = (
        Inf(t) + A2 * p["k21"] + A3 * p["k31"] - A1 * (p["k12"] + p["k13"] + p["k10"])
    )
    ddt_A2 = A1 * p["k12"] - A2 * p["k21"]
    ddt_A3 = A1 * p["k13"] - A3 * p["k31"]
    ddt_Ce = p["ke0"] * (A1 / p["V1"] - Ce)
    return ddt_A1, ddt_A2, ddt_A3, ddt_Ce
