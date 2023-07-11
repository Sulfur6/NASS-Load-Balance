import experiment

if __name__ == '__main__':
    print(experiment.run_single_exp(
        fwd_count=30, fwd_cap_range=[1000, 2000], tenant_count=1000, ins_count_range=[20, 25], ins_cost_range=[10, 100]
    ))
