import torch, json, sys
mass = torch.load(sys.argv[1])[0]; k = int(sys.argv[2])
json.dump([sorted(torch.topk(mass[l], k).indices.tolist()) for l in range(mass.shape[0])], open(sys.argv[3], "w"))
