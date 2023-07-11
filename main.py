import experiment

if __name__ == '__main__':
    print(experiment.run_single_exp(
        fwd_count=40, fwd_cap_range=[1000, 2000], tenant_count=100, ins_count_range=[5, 10], ins_cost_range=[10, 100]
    ))
