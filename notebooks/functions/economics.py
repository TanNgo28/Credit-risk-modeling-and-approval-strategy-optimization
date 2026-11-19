# Step 2: the loan-level economics. Runs once per account-treatment pair.

def amortisation(obj):
    """Outstanding principal path for this contract.

    For an annuity loan the balance after m instalments has a closed form:
        B(m) = A*(1+r)^m - I*((1+r)^m - 1)/r      with r = IR/12
    Amortisation is back-loaded, so the average outstanding balance is well above A/2 --
    which is exactly why the funding cost cannot be approximated as A/2 * rate * years.
    """
    r = obj.p_ir / 12.0
    obj.v_monthly_rate = r

    def balance(m):
        if r == 0:                                       # interest-free: straight-line repayment
            return max(obj.p_amount - obj.p_installment * m, 0.0)
        return max(obj.p_amount * (1 + r) ** m - obj.p_installment * (((1 + r) ** m - 1) / r), 0.0)

    obj.v_balance_fn = balance

    # months paid before a bad loan defaults
    obj.v_months_to_default = max(round(obj.p_tenor * obj.f_default_at), 1)

    # average outstanding over the full tenor, and over the shortened life of a defaulted loan
    n_full = int(obj.p_tenor)
    n_def  = int(obj.v_months_to_default)
    obj.v_avg_outstanding_good = sum(balance(m) for m in range(n_full + 1)) / (n_full + 1)
    obj.v_avg_outstanding_bad  = sum(balance(m) for m in range(n_def + 1)) / (n_def + 1)
    obj.v_outstanding_at_default = balance(n_def)


def profit_good(obj):
    """Profit if the loan performs to term."""
    obj.v_interest_income = obj.p_installment * obj.p_tenor - obj.p_amount
    obj.v_funding_cost_good = (obj.v_avg_outstanding_good * obj.f_cost_of_funds
                               * obj.p_tenor / 12.0)
    obj.v_servicing_good = obj.f_servicing_cost * obj.p_tenor

    obj.v_profit_good = (obj.v_interest_income
                         - obj.v_funding_cost_good
                         - obj.v_servicing_good
                         - obj.f_acq_cost)


def profit_bad(obj):
    """Profit (negative) if the loan defaults.

    Three things must be right:
      1. interest actually collected before default is NOT installment * months --
         part of every instalment is principal repayment
      2. funding and servicing cost run only over the SHORTENED life
      3. the loss is the OUTSTANDING balance at default, net of recovery -- not the ticket
    """
    m_def = obj.v_months_to_default

    # 1. principal repaid before default, hence the interest part of the instalments collected
    principal_repaid = obj.p_amount - obj.v_outstanding_at_default
    obj.v_interest_collected = obj.p_installment * m_def - principal_repaid

    # 2. funding cost over m_def months on the average outstanding of the shortened life
    obj.v_funding_cost_bad = obj.v_avg_outstanding_bad * obj.f_cost_of_funds * m_def / 12.0
    obj.v_servicing_bad = obj.f_servicing_cost * m_def

    # 3. credit loss = outstanding at default * (1 - recovery rate)
    obj.v_credit_loss = obj.v_outstanding_at_default * (1.0 - obj.f_recovery_rate)

    # the acquisition cost is paid on a bad loan too
    obj.v_profit_bad = (obj.v_interest_collected
                        - obj.v_funding_cost_bad
                        - obj.v_servicing_bad
                        - obj.v_credit_loss
                        - obj.f_acq_cost)


def metrics(obj):
    """Expected values, and the treatment switch that makes a rejected loan cost nothing."""
    # expected profit of BOOKING this loan
    _exp_profit_if_booked = (1.0 - obj.p_pd) * obj.v_profit_good + obj.p_pd * obj.v_profit_bad

    # The decision switch. Every constraint variable must be zero for a rejected account,
    # otherwise rejected loans silently contribute to your portfolio aggregates.
    obj.v_approved = obj.a_approve_action
    obj.v_exp_profit = obj.a_approve_action * _exp_profit_if_booked

    # Scaled to millions of VND. PyDO does not rescale coefficients for you, and a matrix with a
    # 1e9 dynamic range is a numerically fragile matrix -- see coefficient_report() in section 7.
    obj.v_disbursed_amount_m = obj.a_approve_action * obj.p_amount / 1_000_000

    # Numerator of the risk-rate constraint, as a weighted-average input (see C1 in section 6):
    # WEIGHTED_AVG(v_pd_disbursed, v_disbursed_amount_m) = sum(PD * disbursed) / sum(disbursed).
    obj.v_pd_disbursed = obj.p_pd

    # expected bad volume, for the report (millions of VND)
    obj.v_exp_bad_volume_m = obj.a_approve_action * obj.p_pd * obj.p_amount / 1_000_000

    # Kept ONLY for the numerical-conditioning experiment of section 7b: the same quantities in raw VND / in millions.
    obj.v_disbursed_amount = obj.a_approve_action * obj.p_amount
    obj.v_exp_profit_m = obj.v_exp_profit / 1_000_000
