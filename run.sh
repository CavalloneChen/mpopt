#Done: python runs/benchmark_opt.py -b CEC13 -d 30 -a LoTFWA -r 50

#Done python runs/benchmark_opt.py -b CEC17 -d 30 -a LoTFWA -r 50

#Done: python runs/benchmark_opt.py -b CEC20 -d 10 -a LoTFWA -r 30

#Done: python runs/benchmark_opt.py -b CEC17 -d 30 -a CMAES -r 50

#Done: python runs/benchmark_opt.py -b CEC20 -d 10 -a CMAES -r 30

#Done python runs/benchmark_opt.py -b CEC20 -d 10 -a CMAFWA -r 30

#Run:
python runs/benchmark_opt.py -b CEC20 -d 10 -a SimpleCMAFWA -r 30

#Run 
#python runs/benchmark_opt.py -b CEC20 -d 10 -a HCFWA -r 30 --name middle --apd "{'fp_method': 'middle'}"
#python runs/benchmark_opt.py -b CEC20 -d 10 -a HCFWA -r 30 --name equal  --apd "{'fp_method': 'equal'}"
#python runs/benchmark_opt.py -b CEC20 -d 10 -a HCFWA -r 30 --name rank   --apd "{'fp_method': 'rank'}"
