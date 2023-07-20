import experiment

if __name__ == '__main__':
    with open('tmp1', 'w') as fw:
        fw.write(str(experiment.repeat_MIC_exp(fwd_count=200, fwd_cap=3000, tenant_count=140,
                                               ins_cost_range=[50, 200], avg_ipc=8)))
