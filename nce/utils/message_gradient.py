import torch, copy


def get_message_gradient(gm, bucket_var, gradient_factors=None, iB = 100):
    from nce.inference.graphical_model import FastGM
    from nce.inference.factor import FastFactor
    from nce.inference.elimination_order import wtminfill_order
    # function should do elimination up to, but not including bucket_var.
    # function should gather all factors from all buckets that come after bucket_var and create a new fastGM from them.
    # function should eliminate all bucket the variables in the scope of bucket_var's bucket's scope
    if gradient_factors is None:
        gm.eliminate_variables(up_to=bucket_var, exact=True)
        gradient_factors = []
        for var in gm.elim_order[gm.elim_order.index(gm.matching_var(bucket_var))+1:]:
            bucket = gm.buckets[var]
            bucket_factors = bucket.factors
            for factor in bucket_factors:
                gradient_factors.append(factor.to_exact())
    bucket = gm.get_bucket(bucket_var)
    message = bucket.compute_message_exact()
    bucket_scope = bucket.get_message_scope()
    if bucket_scope == []:
        return FastFactor(torch.tensor([0.0], device=gm.device, requires_grad=False), []), message
    # if no downstream function
    if gradient_factors == []:
        return FastFactor(torch.tensor([0.0], device=gm.device, requires_grad=False), []), message
    downstream_elim_order = wtminfill_order(gradient_factors, variables_not_eliminated=bucket_scope)
    # print("deo is ", downstream_elim_order)
    # device_copy=str(gm.device)
    downstream_gm = FastGM(factors=gradient_factors, elim_order=downstream_elim_order, reference_fastgm=gm, device=gm.device, nn_config=gm.config)
    downstream_gm.ecl = 2**gm.config['iB_backwards']
    downstream_gm.is_primary = False
    print("Upstream width is ", downstream_gm.get_max_width())
    downstream_gm.iB = iB
    downstream_gm.eliminate_variables(all_but=bucket_scope)
    mg = downstream_gm.get_joint_distribution()
    if mg is None:
        mg = FastFactor(torch.tensor([0.0], device=gm.device, requires_grad=False), [])
    return mg, message

