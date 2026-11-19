# Step 1 of the pipeline: copy what we need out of obj.account / obj.decision / obj.facts
# into short, conventionally-named attributes. Everything downstream reads these.

def flatten(obj):
    # --- raw portfolio fields (p_) ---
    obj.p_amount        = obj.account["AMOUNT_VND"]
    obj.p_tenor         = obj.account["TENOR_MONTHS"]
    obj.p_ir            = obj.account["INTEREST_RATE"]
    obj.p_installment   = obj.account["MONTHLY_INSTALLMENT"]
    obj.p_pd            = obj.account["PD"]
    obj.p_product_group = obj.account["PRODUCT_GROUP"]

    # --- decision field (a_) ---
    obj.a_approve_action = obj.decision["APPROVE_ACTION"]

    # --- facts (f_) ---
    obj.f_cost_of_funds  = obj.facts.get("cost_of_funds")
    obj.f_acq_cost       = obj.facts.get("acquisition_cost")
    obj.f_servicing_cost = obj.facts.get("servicing_cost")
    obj.f_default_at     = obj.facts.get("default_at")
    obj.f_recovery_rate  = obj.facts.get("recovery_rate")
