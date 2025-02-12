using Distributed
nprocs() == 1 && addprocs()
@everywhere using AutoAI
tbsearch()
