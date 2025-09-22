bsub -P acc_DiseaseGeneCell -q premium -n 1 -W 01:00 -R span[hosts=1] -J j1 < ./scripts/runs/runs.lsf c3
